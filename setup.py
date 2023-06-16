#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="1.0",
    description="A transfer learning framework using pytorch, pytorch-lightning and hydra.",
    author="Dennis Ritter",
    author_email="dennis.ritter@bht-berlin.de",
    url="https://gitlab.beuth-hochschule.de/iisy/SynthNet/synthnet-transfer-learning",
    packages=find_packages(),
)
