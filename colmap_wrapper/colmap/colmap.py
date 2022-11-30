#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.

Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

# Built-in/Generic Imports
from pathlib import Path

# Libs
import numpy as np

# Own modules
from colmap_wrapper.colmap.colmap_project import COLMAPProject


class COLMAP(object):
    def __init__(self, project_path: str,
                 dense_pc='fused.ply',
                 load_images: bool = True,
                 load_depth: bool = False,
                 image_resize: float = 1.,
                 bg_color: np.ndarray = np.asarray([1, 1, 1])):

        self.vis_bg_color = bg_color
        self._project_path: Path = Path(project_path)

        if '~' in str(self._project_path):
            self._project_path = self._project_path.expanduser()

        self._sparse_base_path = self._project_path.joinpath('sparse')
        if not self._sparse_base_path.exists():
            raise ValueError('Colmap project structure: sparse folder (cameras, images, points3d) can not be found')

        project_structure = {}
        self._dense_base_path = self._project_path.joinpath('dense')

        # Test if path is a file to get number of subprojects. Single Project with no numeric folder
        if not all([path.is_dir() for path in self._sparse_base_path.iterdir()]):
            project_structure.update({0: {
                "project_path": self._project_path,
                "sparse": self._sparse_base_path,
                "dense": self._dense_base_path}})
        else:
            # In case of folder with reconstruction number after sparse (multiple projects) (e.g. 0,1,2)
            for project_index, sparse_project_path in enumerate(list(self._sparse_base_path.iterdir())):
                project_structure.update({project_index: {"sparse": sparse_project_path}})

            for project_index, dense_project_path in enumerate(list(self._dense_base_path.iterdir())):
                project_structure[project_index].update({"dense": dense_project_path})

            for project_index in project_structure.keys():
                project_structure[project_index].update({"project_path":  self._project_path})

        self.project_list = []
        self.model_ids = []

        for project_index in project_structure.keys():
            self.model_ids.append(project_index)

            project = COLMAPProject(project_path=project_structure[project_index],
                                    dense_pc='fused.ply',
                                    load_images=load_images,
                                    load_depth=load_depth,
                                    image_resize=0.4,
                                    bg_color=bg_color)

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


if __name__ == '__main__':
    from colmap_wrapper.visualization import ColmapVisualization

    MODE = 'multi'

    if MODE == "single":
        from colmap_wrapper.data.download import Dataset

        downloader = Dataset()
        downloader.download_bunny_dataset()

        project = COLMAP(project_path=downloader.file_path,
                         load_images=True,
                         load_depth=True,
                         image_resize=0.4)

        colmap_project = project.projects

        camera = colmap_project.cameras
        images = colmap_project.images
        sparse = colmap_project.get_sparse()
        dense = colmap_project.get_dense()

        project_vs = ColmapVisualization(colmap=colmap_project)
        project_vs.visualization(frustum_scale=0.8, image_type='image')
    elif MODE == "multi":
        project = COLMAP(project_path='/home/luigi/Dropbox/07_data/For5G/22_11_14/reco',
                         dense_pc='fused.ply',
                         load_images=True,
                         load_depth=False,
                         image_resize=0.4)

        for model_idx, COLMAP_MODEL in enumerate(project.projects):
            camera = COLMAP_MODEL.cameras
            images = COLMAP_MODEL.images
            sparse = COLMAP_MODEL.get_sparse()
            dense = COLMAP_MODEL.get_dense()
            project_vs = ColmapVisualization(colmap=COLMAP_MODEL)
            project_vs.visualization(frustum_scale=0.8, image_type='image')
