#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""
import os

from colmap_wrapper.dataloader import *
from colmap_wrapper.reconstruction import *
from colmap_wrapper.visualization import *
from colmap_wrapper.data import *
from colmap_wrapper.gps import *

if not os.getenv("GITHUB_ACTIONS"):
    import getpass

    USER_NAME = getpass.getuser()
