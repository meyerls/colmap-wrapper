#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

# Built-in/Generic Imports
# ...

# Libs
import pathlib as path
import PIL
import cv2

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# Own modules
try:
    from utils import *
    from visualization import *
    from bin import *
except ImportError:
    from .utils import *
    from .visualization import *
    from .bin import *


def load_image(image_path: str) -> np.ndarray:
    """
    Load Image. This takes almost 50% of the time. Would be nice if it is possible to speed up this process. Any
    ideas?

    :param image_path:
    :return:
    """
    return np.asarray(PIL.Image.open(image_path))


class COLMAP:
    def __init__(self, project_path: str,
                 dense_pc: str = 'fused.ply',
                 load_images: bool = False,
                 resize: float = 1.,
                 bg_color: np.ndarray = np.asarray([0.5, 0.5, 0.5])):
        '''
        This is a simple COLMAP project wrapper to simplify the readout of a COLMAP project.
        THE COLMAP project is assumed to be in the following workspace folder structure as suggested in the COLMAP
        documentation (https://colmap.github.io/format.html):

            +── images
            │   +── image1.jpg
            │   +── image2.jpg
            │   +── ...
            +── sparse
            │   +── cameras.txt
            │   +── images.txt
            │   +── points3D.txt
            +── stereo
            │   +── consistency_graphs
            │   │   +── image1.jpg.photometric.bin
            │   │   +── image2.jpg.photometric.bin
            │   │   +── ...
            │   +── depth_maps
            │   │   +── image1.jpg.photometric.bin
            │   │   +── image2.jpg.photometric.bin
            │   │   +── ...
            │   +── normal_maps
            │   │   +── image1.jpg.photometric.bin
            │   │   +── image2.jpg.photometric.bin
            │   │   +── ...
            │   +── patch-match.cfg
            │   +── fusion.cfg
            +── fused.ply
            +── meshed-poisson.ply
            +── meshed-delaunay.ply
            +── run-colmap-geometric.sh
            +── run-colmap-photometric.sh

        :param project_path:
        :param image_path:
        '''
        self.__project_path = path.Path(project_path)
        self.__src_image_path = self.__project_path.joinpath('images')
        self.__sparse_base_path = self.__project_path.joinpath('sparse')
        if not self.__sparse_base_path.exists():
            raise ValueError('Colmap project structure: sparse folder (cameras, images, points3d) can not be found')

        self.__camera_path = self.__sparse_base_path.joinpath('cameras.bin')
        self.__image_path = self.__sparse_base_path.joinpath('images.bin')
        self.__points3D_path = self.__sparse_base_path.joinpath('points3D.bin')
        self.__fused_path = self.__project_path.joinpath(dense_pc)

        self.__load_images = load_images

        self.__read_cameras()
        self.__read_images()
        self.__read_sparse_model()
        self.__read_dense_model()

        self.resize = resize
        self.__add_infos()

        self.vis_bg_color = bg_color

    def __add_infos(self):
        for image_idx in self.images.keys():
            self.images[image_idx].path = self.__src_image_path / self.images[image_idx].name
            if self.__load_images:
                image = load_image(self.images[image_idx].path)

                if self.resize != 1.:
                    image = cv2.resize(image, (0, 0), fx=self.resize, fy=self.resize)

                self.images[image_idx].image = image
            else:
                self.images[image_idx].image = None

    def __read_cameras(self):
        self.cameras = read_cameras_binary(self.__camera_path)

    def __read_images(self):
        self.images = read_images_binary(self.__image_path)

    def __read_sparse_model(self):
        self.sparse = read_points3d_binary(self.__points3D_path)

    def __read_dense_model(self):
        self.dense = o3d.io.read_point_cloud(self.__fused_path.__str__())

    def get_sparse(self):
        return generate_colmap_sparse_pc(self.sparse)

    def show_sparse(self):
        sparse = self.get_sparse()
        o3d.visualization.draw_geometries([sparse])

    def get_dense(self):
        return self.dense

    def show_dense(self):
        dense = self.get_dense()
        o3d.visualization.draw_geometries([dense])

    def visualization(self, frustum_scale: float = 1., point_size: float = 1.):
        """

        :param point_size:
        :param frustum_scale:
        :return:
        """

        geometries = [self.get_dense()]

        for image_idx in self.images.keys():
            camera_intrinsics = Intrinsics(camera=self.cameras[self.images[image_idx].camera_id])

            Rwc, twc, M = convert_colmap_extrinsics(frame=self.images[image_idx])

            line_set, sphere, mesh = draw_camera_viewport(extrinsics=M,
                                                          intrinsics=camera_intrinsics.K,
                                                          image=self.images[image_idx].image,
                                                          scale=frustum_scale)

            geometries.append(mesh)
            geometries.append(line_set)
            geometries.extend(sphere)

        self.geometries = geometries
        self.start_visualizer(geometries=geometries, point_size=point_size)

    def start_visualizer(self, geometries: list,
                         point_size: float,
                         title: str = "Open3D Visualizer",
                         size: tuple = (1920, 1080)):
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=title, width=size[0], height=size[1])

        for geometry in geometries:
            viewer.add_geometry(geometry)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.point_size = point_size
        opt.background_color = self.vis_bg_color
        viewer.run()
        viewer.destroy_window()


if __name__ == '__main__':
    project = COLMAP(project_path='data/door', load_images=True, resize=0.4)

    camera = project.cameras
    images = project.images
    sparse = project.get_sparse()
    dense = project.get_dense()

    project.visualization()

