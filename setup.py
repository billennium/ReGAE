from setuptools import setup, find_packages

setup(
    name="rgae",
    version="1.0.0",
    python_requires=">3.7",
    author_email="adam.malkowski@billennium.com",
    description="R-GAE: Graph autoencoder based on recursive neural networks",
    packages=find_packages(),
    install_requires=[
        "tqdm==4.62.1",
        "pandas==1.3.3",
        "torchmetrics==0.6.1",
        "matplotlib==3.4.3",
        "scipy==1.7.1",
        "networkx==2.6.2",
        "torch==1.9.0",
        "numpy==1.21.2",
        "pytorch_lightning==1.5.6",
        "setuptools==59.5.0"
    ],
)
