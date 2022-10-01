#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

# Built-in/Generic Imports
import os
import pathlib

import pycolmap

# Libs

# Own modules

try:
    from colmap_wrapper.utils import *
    from colmap_wrapper.visualization import *
    from colmap_wrapper.bin import *
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


def __read_depth_image(path):
    depth_map = read_array(path)
    return depth_map
    # Visualization
    # min_depth, max_depth = np.percentile(
    #    depth_map, [5, 95])
    # depth_map[depth_map < min_depth] = min_depth
    # depth_map[depth_map > max_depth] = max_depth
    # from pylab import plt
    # plt.imshow(depth_map)
    # plt.show()


class PhotogrammetrySoftware(object):
    def __init__(self, project_path):
        self._project_path = None

        self.sparse = None
        self.dense = None

        self.geometries = []

    def __read_images(self):
        return NotImplementedError

    def get_sparse(self):
        return generate_colmap_sparse_pc(self.sparse)

    def show_sparse(self):
        o3d.visualization.draw_geometries([self.get_sparse()])

    def get_dense(self):
        return self.dense

    def show_dense(self):
        o3d.visualization.draw_geometries([self.get_dense()])


class COLMAP(PhotogrammetrySoftware):
    def __init__(self, project_path: str,
                 dense_pc: str = 'fused.ply',
                 load_images: bool = False,
                 load_depth: bool = False,
                 image_resize: float = 1.,
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
        PhotogrammetrySoftware.__init__(self, project_path=project_path)

        self._project_path = path.Path(project_path)

        if '~' in str(self._project_path):
            self._project_path = self._project_path.expanduser()

        self._sparse_base_path = self._project_path.joinpath('sparse')
        if not self._sparse_base_path.exists():
            raise ValueError('Colmap project structure: sparse folder (cameras, images, points3d) can not be found')
        if self._sparse_base_path.joinpath('0').exists():
            self._sparse_base_path = self._sparse_base_path.joinpath('0')

        self._dense_base_path = self._project_path.joinpath('dense')
        if self._dense_base_path.joinpath('0').exists():
            self._dense_base_path = self._dense_base_path.joinpath('0')

        if not self._dense_base_path.exists():
            self._dense_base_path = self._project_path

        # Loads undistorted images
        self._src_image_path = self._dense_base_path.joinpath('images')
        self._fused_path = self._dense_base_path.joinpath(dense_pc)
        self._stereo_path = self._dense_base_path.joinpath('stereo')
        self._depth_image_path = self._stereo_path.joinpath('depth_maps')
        self._normal_image_path = self._stereo_path.joinpath('normal_maps')

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

        self.load_images = load_images
        self.load_depth = load_depth
        self.image_resize = image_resize
        self.vis_bg_color = bg_color

        self.read()

    def read(self):
        self.__read_cameras()
        self.__read_images()
        self.__read_sparse_model()
        self.__read_dense_model()
        self.__read_depth_structure()
        self.__add_infos()

    def __add_infos(self):
        self.max_depth_scaler = 0
        for image_idx in self.images.keys():
            self.images[image_idx].path = self._src_image_path / self.images[image_idx].name
            if self.load_images:
                image = load_image(self.images[image_idx].path)

                if self.image_resize != 1.:
                    image = cv2.resize(image, (0, 0), fx=self.image_resize, fy=self.image_resize)

                self.images[image_idx].image = image
            else:
                self.images[image_idx].image = None

            if self.load_depth:
                self.images[image_idx].depth_image_geometric = read_array(
                    path=next((p for p in self.depth_path_geometric if self.images[image_idx].name in p), None))

                # print(self.images[image_idx].name)
                # print(next((p for p in self.depth_path_geometric if self.images[image_idx].name in p), None))
                # print('\n')

                min_depth, max_depth = np.percentile(self.images[image_idx].depth_image_geometric, [5, 95])

                if max_depth > self.max_depth_scaler:
                    self.max_depth_scaler = max_depth

                self.images[image_idx].depth_image_photometric = read_array(
                    path=next((p for p in self.depth_path_photometric if self.images[image_idx].name in p), None))
            else:
                self.images[image_idx].depth_image_geometric = None
                self.images[image_idx].depth_path_photometric = None
            # self.images[image_idx].normal_image = self.__read_depth_images

            self.images[image_idx].intrinsics = Intrinsics(camera=self.cameras[self.images[image_idx].camera_id])

            # Fixing Strange Error when cy is negative
            if self.images[image_idx].intrinsics.cx < 0:
                pass

            if self.images[image_idx].intrinsics.cy < 0:
                pass

    def __read_cameras(self):
        reconstruction = pycolmap.Reconstruction(self._sparse_base_path)
        self.cameras = {}
        for camera_id, camera in reconstruction.cameras.items():
            if camera.model_name == 'SIMPLE_RADIAL':
                params = np.asarray([camera.params[0],  # fx
                                     camera.params[0],  # fy
                                     camera.params[1],  # cx
                                     camera.params[2],  # cy
                                     camera.params[3]])  # k1
                # cv2.getOptimalNewCameraMatrix(camera.calibration_matrix(), [k, 0, 0, 0], (camera.width, camera.height), )

            elif camera.model_name == 'PINHOLE':
                params = np.asarray([camera.params[0],  # fx
                                     camera.params[1],  # fy
                                     camera.params[2],  # cx
                                     camera.params[3],  # cy
                                     0])  # k1

            else:
                raise NotImplementedError('Model {} is not implemented!'.format(model_name))

            camera_params = Camera(id=camera.camera_id,
                                   model=camera.model_name,
                                   width=camera.width,
                                   height=camera.height,
                                   params=params)

            self.cameras.update({camera_id: camera_params})
            # if self._camera_path.suffix == '.txt':
            #    self.cameras = read_cameras_text(self._camera_path)
            # elif self._camera_path.suffix == '.bin':
            # self.cameras = read_cameras_binary(self._camera_path)
            # else:
            #    raise FileNotFoundError('Wrong extension found, {}'.format(self._camera_path.suffix))

    def __read_images(self):
        if self._image_path.suffix == '.txt':
            self.images = read_images_text(self._image_path)
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

    def __read_depth_structure(self):

        self.depth_path_geometric = []
        self.depth_path_photometric = []

        for depth_path in list(self._depth_image_path.glob('*.bin')):
            if 'geometric' in depth_path.__str__():
                self.depth_path_geometric.append(depth_path.__str__())
            elif 'photometric' in depth_path.__str__():
                self.depth_path_photometric.append(depth_path.__str__())
            else:
                raise ValueError('Unkown depth image type: {}'.format(path))

    def __read_dense_model(self):
        self.dense = o3d.io.read_point_cloud(self._fused_path.__str__())

    def add_colmap_dense2geometrie(self):
        self.geometries.append(self.get_dense())

    def add_colmap_sparse2geometrie(self):
        self.geometries.append(self.get_sparse())

    def add_colmap_frustums2geometrie(self, frustum_scale: float = 1., image_type: str = 'image'):
        """

        @param image_type:
        @type frustum_scale: object
        """
        geometries = []
        for image_idx in self.images.keys():

            if image_type == 'image':
                image = self.images[image_idx].image
            elif image_type == 'depth_geo':
                image = self.images[image_idx].depth_image_geometric
                min_depth, max_depth = np.percentile(
                    image, [5, 95])
                image[image < min_depth] = min_depth
                image[image > max_depth] = max_depth
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = (image / self.max_depth_scaler * 255).astype(np.uint8)
                image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
            elif image_type == 'depth_photo':
                image = self.images[image_idx].depth_path_photometric
                min_depth, max_depth = np.percentile(
                    image, [5, 95])
                image[image < min_depth] = min_depth
                image[image > max_depth] = max_depth
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            line_set, sphere, mesh = draw_camera_viewport(extrinsics=self.images[image_idx].extrinsics,
                                                          intrinsics=self.images[image_idx].intrinsics.K,
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
    # project = COLMAP(project_path='data/door', load_images=True, load_depth=True, image_resize=0.4)
    # project = COLMAP(project_path='/home/luigi/Dropbox/07_data/CherrySLAM/test_sequences/01_easy/reco/',
    project = COLMAP(project_path='/home/se86kimy/Dropbox/07_data/misc/bunny_data/reco_DocSem2',
                     dense_pc='fused.ply',
                     load_images=True,
                     load_depth=True,
                     image_resize=0.4)

    camera = project.cameras
    images = project.images
    sparse = project.get_sparse()
    dense = project.get_dense()

    project.visualization(frustum_scale=0.2, image_type='depth_geo', object=1)
