#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup
import os
with open("README.md") as f:
    readme = f.read()

setup(
    name="hlp",
    version="1.0.0",
    description="Hyperlink-induced Pretraining Toolkit",
    url="https://github.com/jzhoubu/HLP/",
    long_description=readme,
    long_description_content_type="text/markdown",
    setup_requires=[
        "setuptools>=18.0",
    ],
    install_requires=[
        "faiss-cpu>=1.6.1",
        "filelock",
        "numpy",
        "regex",
        "torch==1.8.1",
        "transformers==3.0.2",
        "tqdm>=4.27",
        "wget",
        "spacy>=2.1.8",
        "hydra-core>=1.0.0",
        "omegaconf>=2.0.1",
        "jsonlines",
    ],
)

os.system("python -m spacy download en_core_web_sm")