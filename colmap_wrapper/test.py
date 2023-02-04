#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Own Modules
from colmap_wrapper.data.download import Dataset
from colmap_wrapper.colmap.colmap import COLMAP
from colmap_wrapper.visualization import ColmapVisualization

if __name__ == '__main__':
    downloader = Dataset()
    downloader.download_bunny_dataset()

    project = COLMAP(project_path='/home/luigi/Dropbox/07_data/COLMAP_BAPTIST/reco',
                     load_depth=True,
                     image_resize=0.4)

    colmap_project = project.project

    camera = colmap_project.cameras
    images = colmap_project.images
    sparse = colmap_project.get_sparse()
    dense = colmap_project.get_dense()

    project_vs = ColmapVisualization(colmap=colmap_project)
    project_vs.visualization(frustum_scale=0.8, image_type='image')

    print('Finished')
