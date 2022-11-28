#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import open3d as o3d
import numpy as np

from colmap_wrapper.colmap import COLMAP
from colmap_wrapper.colmap.colmap_project import PhotogrammetrySoftware
from colmap_wrapper.visualization import draw_camera_viewport


class PhotogrammetrySoftwareVisualization():
    def __init__(self, photogrammetry_software: PhotogrammetrySoftware):
        self.photogrammetry_software = photogrammetry_software

        self.geometries = []

    def show_sparse(self):
        o3d.visualization.draw_geometries([self.photogrammetry_software.get_sparse()])

    def show_dense(self):
        o3d.visualization.draw_geometries([self.photogrammetry_software.get_dense()])


class ColmapVisualization(PhotogrammetrySoftwareVisualization):
    def __init__(self, colmap: COLMAP, bg_color: np.ndarray = np.asarray([1, 1, 1])):
        super().__init__(colmap)

        self.vis_bg_color = bg_color

    def add_colmap_dense2geometrie(self):
        if np.asarray(self.photogrammetry_software.get_dense().points).shape[0] == 0:
            return False

        self.geometries.append(self.photogrammetry_software.get_dense())

        return True

    def add_colmap_sparse2geometrie(self):
        if np.asarray(self.photogrammetry_software.get_sparse().points).shape[0] == 0:
            return False

        self.geometries.append(self.photogrammetry_software.get_sparse())
        return True

    def add_colmap_frustums2geometrie(self, frustum_scale: float = 1., image_type: str = 'image'):
        """
        @param image_type:
        @type frustum_scale: object
        """
        import cv2

        geometries = []
        for image_idx in self.photogrammetry_software.images.keys():

            if image_type == 'image':
                image = self.photogrammetry_software.images[image_idx].getData(
                    self.photogrammetry_software.image_resize)
            elif image_type == 'depth_geo':
                image = self.photogrammetry_software.images[image_idx].depth_image_geometric
                min_depth, max_depth = np.percentile(image, [5, 95])
                image[image < min_depth] = min_depth
                image[image > max_depth] = max_depth
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = (image / self.max_depth_scaler * 255).astype(np.uint8)
                image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
            elif image_type == 'depth_photo':
                image = self.photogrammetry_software.images[image_idx].depth_image_photometric
                min_depth, max_depth = np.percentile(
                    image, [5, 95])
                image[image < min_depth] = min_depth
                image[image > max_depth] = max_depth
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            line_set, sphere, mesh = draw_camera_viewport(
                extrinsics=self.photogrammetry_software.images[image_idx].extrinsics,
                intrinsics=self.photogrammetry_software.images[image_idx].intrinsics.K,
                image=image,
                scale=frustum_scale)

            geometries.append(mesh)
            geometries.append(line_set)
            geometries.extend(sphere)

        self.geometries.extend(geometries)

    def visualization(self, frustum_scale: float = 1., point_size: float = 1., image_type: str = 'image', *args):
        """
        @param frustum_scale:
        @param point_size:
        @param image_type: ['image, depth_geo', 'depth_photo']
        """
        image_types = ['image', 'depth_geo', 'depth_photo']

        if image_type not in image_types:
            raise TypeError('image type is {}. Only {} is allowed'.format(image_type, image_types))

        self.add_colmap_dense2geometrie()
        self.add_colmap_sparse2geometrie()
        self.add_colmap_frustums2geometrie(frustum_scale=frustum_scale, image_type=image_type)
        self.start_visualizer(point_size=point_size)

    def start_visualizer(self,
                         point_size: float,
                         title: str = "Open3D Visualizer",
                         size: tuple = (1920, 1080)):
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=title, width=size[0], height=size[1])

        for geometry in self.geometries:
            viewer.add_geometry(geometry)
        opt = viewer.get_render_option()
        # opt.show_coordinate_frame = True
        opt.point_size = point_size
        opt.line_width = 0.01
        opt.background_color = self.vis_bg_color
        viewer.run()
        viewer.destroy_window()


if __name__ == '__main__':
    project = COLMAP(project_path='/home/luigi/Dropbox/07_data/misc/bunny_data/reco_DocSem2',
                     dense_pc='fused.ply',
                     load_images=True,
                     image_resize=0.4)

    project_vs = ColmapVisualization(colmap=project.project_list[0])
    project_vs.visualization(frustum_scale=0.8, image_type='image')
