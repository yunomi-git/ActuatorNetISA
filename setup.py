from setuptools import find_packages, setup

setup(
    name="actuatorNet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "scikit-learn",
        "matplotlib",
        "wandb",
        "tqdm",
        "pyyaml"
    ]
)