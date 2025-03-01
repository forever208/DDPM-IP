from setuptools import setup

setup(
    name="guided-diffusion",
    py_modules=["guided_diffusion"],
    install_requires=["blobfile==2.0.2", "torch==1.13.0", "tqdm==4.65.0"],
)
