#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import open3d as o3d
import numpy as np
from tqdm import tqdm

from colmap_wrapper.colmap import COLMAP
from colmap_wrapper.colmap.colmap_project import PhotogrammetrySoftware
from colmap_wrapper.visualization import draw_camera_viewport


class PhotogrammetrySoftwareVisualization():
    def __init__(self, photogrammetry_software: PhotogrammetrySoftware, image_resize: float = 0.3):
        self.photogrammetry_software = photogrammetry_software
        self.geometries = []

        self.image_resize: float = image_resize

    def show_sparse(self):
        o3d.visualization.draw_geometries([self.photogrammetry_software.get_sparse()])

    def show_dense(self):
        o3d.visualization.draw_geometries([self.photogrammetry_software.get_dense()])


class ColmapVisualization(PhotogrammetrySoftwareVisualization):
    def __init__(self, colmap: COLMAP, bg_color: np.ndarray = np.asarray([1, 1, 1]), image_resize: float = 0.3):
        super().__init__(colmap, image_resize=image_resize)

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
        # aa = []
        # for image_idx in tqdm(self.photogrammetry_software.images.keys()):
        #    aa.append(self.photogrammetry_software.images[image_idx].extrinsics[:3, 3])
        #
        # a = o3d.geometry.PointCloud()
        # a.points = o3d.utility.Vector3dVector(np.vstack(aa))
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=np.asarray([0., 0., 0.]))
        # o3d.visualization.draw_geometries([coordinate_frame, a])

        geometries = []
        for image_idx in tqdm(self.photogrammetry_software.images.keys()):

            if image_type == 'image':
                image = self.photogrammetry_software.images[image_idx].getData(self.image_resize)
            elif image_type == 'depth_geo':
                import cv2
                image = self.photogrammetry_software.images[image_idx].depth_image_geometric
                min_depth, max_depth = np.percentile(image, [5, 95])
                image[image < min_depth] = min_depth
                image[image > max_depth] = max_depth
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = (image / self.photogrammetry_software.max_depth_scaler * 255).astype(
                    np.uint8)  # TODO max_depth_scaler
                image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
            elif image_type == 'depth_photo':
                import cv2
                image = self.photogrammetry_software.images[image_idx].depth_image_photometric
                min_depth, max_depth = np.percentile(
                    image, [5, 95])
                image[image < min_depth] = min_depth
                image[image > max_depth] = max_depth
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = (image / self.photogrammetry_software.max_depth_scaler_photometric * 255).astype(
                    np.uint8)  # TODO max_depth_scaler_photometric

            line_set, sphere, mesh = draw_camera_viewport(
                extrinsics=self.photogrammetry_software.images[image_idx].extrinsics,
                intrinsics=self.photogrammetry_software.images[image_idx].intrinsics.K,
                image=image,
                scale=frustum_scale)

            geometries.append(mesh)
            geometries.append(line_set)
            geometries.extend(sphere)

        self.geometries.extend(geometries)

    def visualization(self, frustum_scale: float = 1., point_size: float = 1., image_type: str = 'image',
                      title: str = "Open3D Visualizer", window_size: tuple = (1920, 1080)):
        """
        @param frustum_scale:
        @param point_size:
        @param image_type: ['image, depth_geo', 'depth_photo']
        @param title:
        @param window_size:
        """
        image_types = ['image', 'depth_geo', 'depth_photo']

        if image_type not in image_types:
            raise TypeError('image type is {}. Only {} is allowed'.format(image_type, image_types))

        self.add_colmap_dense2geometrie()
        self.add_colmap_sparse2geometrie()
        self.add_colmap_frustums2geometrie(frustum_scale=frustum_scale, image_type=image_type)
        self.start_visualizer(point_size=point_size, title=title, size=window_size)

    def start_visualizer(self,
                         point_size: float,
                         title: str = "Open3D Visualizer",
                         size: tuple = (1920, 1080)):
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=title, width=size[0], height=size[1])

        for geometry in self.geometries:
            viewer.add_geometry(geometry)
        opt = viewer.get_render_option()
        ctr = viewer.get_view_control()
        ctr.set_front([-0.25053090455707444, -0.86588036871950691, 0.43299590405451305])
        ctr.set_lookat([6.298907877205747, 1.3968597508640934, 1.9543917296138904])
        ctr.set_up([0.06321823212762033, -0.460937346978886, -0.88517807094771861])
        ctr.set_zoom(0.02)
        # opt.show_coordinate_frame = True
        opt.point_size = point_size
        opt.light_on = False
        opt.line_width = 0.01
        opt.background_color = self.vis_bg_color

        viewer.capture_screen_image(filename='./test.png', do_render=True)
        viewer.run()
        viewer.destroy_window()


if __name__ == '__main__':
    project = COLMAP(project_path='/home/luigi/Dropbox/07_data/misc/bunny_data/reco_DocSem2',
                     dense_pc='fused.ply')

    project_vs = ColmapVisualization(colmap=project.project_list[0], image_resize=0.4)
    project_vs.visualization(frustum_scale=0.8, image_type='image')
