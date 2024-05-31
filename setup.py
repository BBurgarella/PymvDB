# setup.py
from setuptools import setup, find_packages

setup(
    name="image_embedding_lib",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "Pillow",
        "transformers",
        "torch",
    ],
    author="Boris Burgarella",
    author_email="b.burgarella@gmail.com",
    description="A simple vector database for images",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/BBurgarella/PymvDB"
)
