import copy
import functools
import os
import time

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.num_iter = 0
        self.pred_error = {}
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        # resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        # if resume_checkpoint:
        #     self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
        #     # if dist.get_rank() == 0:
        #     logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
        #     self.model.load_state_dict(
        #         dist_util.load_state_dict(
        #             resume_checkpoint, map_location=dist_util.dev()
        #         )
        #     )
        logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")
        self.model.load_state_dict(dist_util.load_state_dict(self.resume_checkpoint, map_location=dist_util.dev()))
        logger.log(f"checkpoint loaded...")

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # if dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (self.num_iter < 100):
            logger.log(f" ")
            logger.log(f"computing {self.num_iter+1} batch...")
            batch, cond = next(self.data)

            # compute prediction error for each timestep
            for t in range(0, 1000, 10):  # [0, 1, 2, ...999]
                pred_error = self.run_step(batch, cond, t)  # 4D array (batch, 3, 32, 32)

                if str(t) in self.pred_error.keys():
                    self.pred_error[str(t)] = np.concatenate((self.pred_error[str(t)], pred_error), axis=0)
                else:
                    self.pred_error[str(t)] = pred_error

            self.num_iter += 1
            # logger.log(f"number of error samples for Gaussian visualization: {len(self.pred_error['2'])}")
            logger.log(f"current array shape of each t: {self.pred_error['0'].shape}")

        logger.log(f" ")
        logger.log(f"saving data as npz...")
        all_t = list(self.pred_error.keys())
        npz = np.stack((self.pred_error[all_t[0]], self.pred_error[all_t[1]]))  # 5D array (2, batch, 3, 32, 32)
        for t in all_t[2:]:
            npz = np.concatenate((npz, self.pred_error[t][np.newaxis, :]), axis=0)

        logger.log(f"npz size: {npz.shape}")  # 5D array (100, batch, 3, 32, 32)
        path = './imagenet32_base_gaussian_error/gaussian_error_xstart_all_pixels'
        np.savez(path, npz)
        logger.log(f"array saved into {path}")

    def run_step(self, batch, cond, t):
        return self.forward_backward(batch, cond, t)
        # took_step = self.mp_trainer.optimize(self.opt)
        # if took_step:
        #     self._update_ema()
        # self._anneal_lr()
        # self.log_step()

    def forward_backward(self, batch, cond, t):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]

            # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            ones = th.ones(micro.shape[0]).long().to(dist_util.dev())
            timestep = ones * t

            # sample x_t
            forward_fn = self.diffusion.sample_x_t
            gt_x_t, _ = forward_fn(
                micro,
                timestep,
                model_kwargs=micro_cond,
            )

            # run inference to get eps_hat
            sample_fn = self.diffusion.p_mean_variance
            predction = sample_fn(
                self.ddp_model,
                gt_x_t,
                timestep,
            )
            pred_x_0 = predction["pred_xstart"]  # predicted x_0

            # compute error between gt_x_0 and pred_x_0 for each pixel
            error = pred_x_0 - micro  # 4D tensor (batch, 3, 32, 32)
            # error = error[:, 1, 15, 15]  # 1D tensor (batch)
            pred_error = error.detach().cpu().numpy()

            return pred_error

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("total batch size", self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
