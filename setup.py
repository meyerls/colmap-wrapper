#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
import setuptools

#with open("Readme.md", 'r') as f:
#    long_description = f.read()

setuptools.setup(
    name='colmap_wrapper',
    version='1.1.2',
    description='COLMAP Handler',
    license="MIT",
    long_description='',
    long_description_content_type="text/markdown",
    author='Lukas Meyer',
    author_email='lukas.meyer@fau.de',
    url="https://github.com/meyerls/colmap-wrapper",
    packages=['colmap_wrapper'],
    install_requires=["numpy"],  # external packages as dependencies
    classifiers=[
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
