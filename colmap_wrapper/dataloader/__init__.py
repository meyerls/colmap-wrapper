#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

from colmap_wrapper.dataloader.utils import *
from colmap_wrapper.dataloader.bin import *
from colmap_wrapper.dataloader.camera import *
from colmap_wrapper.dataloader.loader import *
from colmap_wrapper.dataloader.project import *


## Deprecated
from pathlib import Path
import numpy as np


class COLMAP(object):
    def __init__(
        self,
        project_path: str,
        dense_pc="fused.ply",
        load_images: bool = True,
        load_depth: bool = False,
        image_resize: float = 1.0,
        bg_color: np.ndarray = np.asarray([1, 1, 1]),
        exif_read=False,
    ):
        self.exif_read = exif_read
        self.vis_bg_color = bg_color
        self._project_path: Path = Path(project_path)

        if "~" in str(self._project_path):
            self._project_path = self._project_path.expanduser()

        self._sparse_base_path = self._project_path.joinpath("sparse")
        if not self._sparse_base_path.exists():
            raise ValueError(
                "Colmap project structure: sparse folder (cameras, images, points3d) can not be found"
            )

        project_structure = {}
        self._dense_base_path = self._project_path.joinpath("dense")

        # Test if path is a file to get number of subprojects. Single Project with no numeric folder
        if not all([path.is_dir() for path in self._sparse_base_path.iterdir()]):
            project_structure.update(
                {
                    0: {
                        "project_path": self._project_path,
                        "sparse": self._sparse_base_path,
                        "dense": self._dense_base_path,
                    }
                }
            )
        else:  # In case of folder with reconstruction number after sparse (multiple projects) (e.g. 0,1,2)
            for project_index, sparse_project_path in enumerate(
                list(self._sparse_base_path.iterdir())
            ):
                project_structure.update(
                    {project_index: {"sparse": sparse_project_path}}
                )

            for project_index, dense_project_path in enumerate(
                list(self._dense_base_path.iterdir())
            ):
                project_structure[project_index].update({"dense": dense_project_path})

            for project_index in project_structure.keys():
                project_structure[project_index].update(
                    {"project_path": self._project_path}
                )

        self.project_list = []
        self.model_ids = []

        for project_index in project_structure.keys():
            self.model_ids.append(project_index)

            project = COLMAPProject(
                project_path=project_structure[project_index],
                dense_pc=dense_pc,
                load_images=load_images,
                load_depth=load_depth,
                image_resize=0.4,
                bg_color=bg_color,
                exif_read=self.exif_read,
            )

            self.project_list.append(project)

    @property
    def projects(self):
        if len(self.project_list) == 1:
            return self.project_list[0]
        elif len(self.project_list) > 1:
            return self.project_list

    @projects.setter
    def projects(self, projects):
        self.project_list = projects
