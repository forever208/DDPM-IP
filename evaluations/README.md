# Evaluations

To compare different generative models, we use FID, sFID, Precision, Recall, and Inception Score. These metrics can all be calculated using batches of samples, which we store in `.npz` (numpy) files.

# Solution to Tensorflow2 bug
you might encounter the bug: Could not load dynamic library 'libcudart.so.11.0 etc.

recommended installation:
```
$ pip install tensorflow==2.4
$ conda install cudatoolkit=11.0
$ conda install -c conda-forge cudnn
```

(refer to [libcudart.so.11.0](https://github.com/tensorflow/tensorflow/issues/45930) and [libcudart.so.8](https://github.com/tensorflow/tensorflow/issues/45200) for details)


# Run evaluations

First, generate the reference batch `npz` sample using corresponding python script from the folder `DDPM-IP
/datasets/`. For example, you should pack the whole cifar-10 training dataset and save it into `cifar_train.npz` file before computing the FID.

In the following case, we compute the FID on cifar10, so the refernce batch is `cifar10_train.npz` and we can use the sample batch `samples_50000x32x32x3.npz`.

Next, run the `evaluator.py` script. The requirements of this script can be found in [requirements.txt](requirements.txt). Pass two arguments to the script: the reference batch and the sample batch. The script will download the InceptionV3 model used for evaluations into the current working directory (if it is not already present). This file is roughly 100MB.

The output of the script will look something like this, where the first `...` is a bunch of verbose TensorFlow logging:

```
$ python evaluator.py cifar10_train.npz samples_50000x32x32x3.npz
...
computing reference batch activations cifar10_train.npz...
computing/reading reference batch statistics...
computing sample batch activations samples_50000x32x32x3.npz...
computing/reading sample batch statistics...
Computing evaluations...
Inception Score: 9.700922012329102
FID: 2.3776688254045553
sFID: 4.09995402650668
Precision: 0.69118
Recall: 0.60512
```
