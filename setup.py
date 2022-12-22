from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="qW-Map",
    version="0.1.1",
    description="A PyTorch implementation of Quantum Weight Re-Mapping",
    author="Michael Kölle",
    author_email="michael.koelle@ifi.lmu.de",
    url="https://github.com/michaelkoelle/quantum-weight-remapping",
    license="MIT",
    keywords=[
        "quantum artificial intelligence",
        "pytorch",
        "quantum machine learning",
        "weight re-mapping",
        "quantum supervised learning",
        "quantum variational circuit",
        "quantum variational classifier",
        "pennylane",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["examples"]),
    install_requires=["torch>=1.6"],
)
