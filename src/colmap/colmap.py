#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

# Built-in/Generic Imports
import os

# Libs

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
                 bg_color: np.ndarray = np.asarray([1, 1, 1])):
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
        self._project_path = path.Path(project_path)

        self._src_image_path = self._project_path.joinpath('images')
        if not os.path.exists(self._src_image_path):
            self._src_image_path = self._project_path.joinpath('dense').joinpath('0').joinpath('images')
        self._sparse_base_path = self._project_path.joinpath('sparse')
        if not self._sparse_base_path.exists():
            raise ValueError('Colmap project structure: sparse folder (cameras, images, points3d) can not be found')

        if os.path.exists(self._sparse_base_path.joinpath('0')):
            self._sparse_base_path = self._sparse_base_path.joinpath('0')

        files = []
        types = ('*.txt', '*.bin')
        for t in types:
            files.extend(self._sparse_base_path.glob(t))

        for file_path in files:
            if 'cameras' in file_path.name:
                self._camera_path = file_path
            elif 'images' in file_path.name:
                self._image_path = file_path
            elif 'points3D' in file_path.name:
                self._points3D_path = file_path
            else:
                raise ValueError('Unkown file in sparse folder')

        self._fused_path = self._project_path.joinpath(dense_pc)
        if not os.path.exists(self._fused_path):
            self._fused_path = self._project_path.joinpath('dense').joinpath('0').joinpath(dense_pc)

        self.load_images = load_images
        self.geometries = None
        self.resize = resize
        self.vis_bg_color = bg_color

        self.read()

    def read(self):
        self.__read_cameras()
        self.__read_images()
        self.__read_sparse_model()
        self.__read_dense_model()
        self.__add_infos()

    def __add_infos(self):
        for image_idx in self.images.keys():
            self.images[image_idx].path = self._src_image_path / self.images[image_idx].name
            if self.load_images:
                image = load_image(self.images[image_idx].path)

                if self.resize != 1.:
                    image = cv2.resize(image, (0, 0), fx=self.resize, fy=self.resize)

                self.images[image_idx].image = image
            else:
                self.images[image_idx].image = None

            self.images[image_idx].intrinsics = Intrinsics(camera=self.cameras[self.images[image_idx].camera_id])

    def __read_cameras(self):
        if self._camera_path.suffix == '.txt':
            self.cameras = read_cameras_text(self._camera_path)
        elif self._camera_path.suffix == '.bin':
            self.cameras = read_cameras_binary(self._camera_path)
        else:
            raise FileNotFoundError('Wrong extension found, {}'.format(self._camera_path.suffix))

    def __read_images(self):
        if self._image_path.suffix == '.txt':
            self.images  = read_images_text(self._image_path)
        elif self._image_path.suffix == '.bin':
            self.images = read_images_binary(self._image_path)
        else:
            raise FileNotFoundError('Wrong extension found, {}'.format(self._camera_path.suffix))

    def __read_sparse_model(self):
        if self._points3D_path.suffix == '.txt':
            self.sparse = read_points3D_text(self._points3D_path)
        elif self._points3D_path.suffix == '.bin':
            self.sparse = read_points3d_binary(self._points3D_path)
        else:
            raise FileNotFoundError('Wrong extension found, {}'.format(self._camera_path.suffix))

    def __read_dense_model(self):
        self.dense = o3d.io.read_point_cloud(self._fused_path.__str__())

    def get_sparse(self):
        return generate_colmap_sparse_pc(self.sparse)

    def show_sparse(self):
        o3d.visualization.draw_geometries([self.get_sparse()])

    def get_dense(self):
        return self.dense

    def show_dense(self):
        o3d.visualization.draw_geometries([self.get_dense()])

    def add_colmap_reconstruction_geometries(self, frustum_scale: float = 1., ):
        geometries = [self.get_dense(), self.get_sparse()]

        for image_idx in self.images.keys():
            line_set, sphere, mesh = draw_camera_viewport(extrinsics=self.images[image_idx].extrinsics,
                                                          intrinsics=self.images[image_idx].intrinsics.K,
                                                          image=self.images[image_idx].image,
                                                          scale=frustum_scale)

            geometries.append(mesh)
            geometries.append(line_set)
            geometries.extend(sphere)

        self.geometries = geometries

    def visualization(self, frustum_scale: float = 1., point_size: float = 1.):
        """

        :param point_size:
        :param frustum_scale:
        :return:
        """

        self.add_colmap_reconstruction_geometries(frustum_scale)
        self.start_visualizer(point_size=point_size)

    def start_visualizer_scaled(self,
                                geometries,
                                point_size: float,
                                title: str = "Open3D Visualizer",
                                size: tuple = (1920, 1080)):
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=title, width=size[0], height=size[1])

        for geometry in geometries:
            viewer.add_geometry(geometry)
        opt = viewer.get_render_option()
        # opt.show_coordinate_frame = True
        opt.point_size = point_size
        opt.background_color = self.vis_bg_color
        viewer.run()
        viewer.destroy_window()

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
        opt.background_color = self.vis_bg_color
        viewer.run()
        viewer.destroy_window()

    def write(self, data):
        pass


if __name__ == '__main__':
    project = COLMAP(project_path='data/door', load_images=True, resize=0.4)

    camera = project.cameras
    images = project.images
    sparse = project.get_sparse()
    dense = project.get_dense()

    project.visualization()
