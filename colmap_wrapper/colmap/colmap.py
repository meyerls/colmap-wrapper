#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

import numpy as np
import open3d as o3d
import pycolmap

from pathlib import Path
from colmap_wrapper.colmap import (Camera, Intrinsics, read_array, read_images_text, read_points3D_text, read_points3d_binary, read_images_binary, generate_colmap_sparse_pc)


def __read_depth_image(path):
    depth_map = read_array(path)
    return depth_map

class PhotogrammetrySoftware(object):
    def __init__(self, project_path):
        self._project_path = project_path

        self.sparse = None
        self.dense = None

    def __read_images(self):
        return NotImplementedError

    def get_sparse(self):
        return generate_colmap_sparse_pc(self.sparse)

    def get_dense(self):
        return self.dense

class COLMAP(PhotogrammetrySoftware):
    def __init__(
                    self,
                    project_path: str,
                    dense_pc: str = 'fused.ply',
                    load_depth: bool = False,
                    image_resize: float = 1.
                ):
        """
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

        @param project_path: path to colmap project
        @param dense_pc: path to dense point cloud (Might be useful if pc has been renamed or deviades from fused.ply)
        @param load_depth: flag to load depth images.
        @param image_resize: float to scale images if image size is to large for RAM storage
        @param bg_color: background color for visualization
        """

        PhotogrammetrySoftware.__init__(self, project_path=project_path)

        # Search and Set Paths
        self._project_path = Path(project_path)

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

        # Set Variables
        self.load_depth = load_depth
        self.image_resize = image_resize
        
        # Start Reading Files
        self.read()

    def read(self):
        """
        Start reading all necessary information

        @return:
        """
        self.__read_cameras()
        self.__read_images()
        self.__read_sparse_model()
        self.__read_dense_model()
        self.__read_depth_structure()
        self.__add_infos()

    def __add_infos(self):
        """
        Loads depth images from path and adds it to the Image object.

        @warning: this might exceed your storage! Think about adjusting the scaling parameter.

        @return:
        """
        self.max_depth_scaler = 0
        self.max_depth_scaler_photometric = 0
        for image_idx in self.images.keys():
            self.images[image_idx].path = self._src_image_path / self.images[image_idx].name

            if self.load_depth:
                self.images[image_idx].depth_image_geometric = read_array(path=next((p for p in self.depth_path_geometric if self.images[image_idx].name in p), None))

                _, max_depth = np.percentile(self.images[image_idx].depth_image_geometric, [5, 95])

                if max_depth > self.max_depth_scaler:
                    self.max_depth_scaler = max_depth

                self.images[image_idx].depth_image_photometric = read_array(path=next((p for p in self.depth_path_photometric if self.images[image_idx].name in p), None))

                _, max_depth = np.percentile(self.images[image_idx].depth_image_photometric, [5, 95])

                if max_depth > self.max_depth_scaler_photometric:
                    self.max_depth_scaler_photometric = max_depth

            else:
                self.images[image_idx].depth_image_geometric = None
                self.images[image_idx].depth_path_photometric = None

            self.images[image_idx].intrinsics = Intrinsics(camera=self.cameras[self.images[image_idx].camera_id])

            # Fixing Strange Error when cy is negative
            if self.images[image_idx].intrinsics.cx < 0:
                pass

            if self.images[image_idx].intrinsics.cy < 0:
                pass

    def __read_cameras(self):
        """
        Load camera model from file. Currently only Simple Radial and 'Pinhole' are supported. If the camera settings
        are identical for all images only one camera is provided. Otherwise, every image has its own camera model.

        @return:
        """
        reconstruction = pycolmap.Reconstruction(self._sparse_base_path)
        self.cameras = {}
        for camera_id, camera in reconstruction.cameras.items():
            if camera.model_name == 'SIMPLE_RADIAL':
                params = np.asarray([camera.params[0],  # fx
                                     camera.params[0],  # fy
                                     camera.params[1],  # cx
                                     camera.params[2],  # cy
                                     camera.params[3]])  # k1

            elif camera.model_name == 'PINHOLE':
                params = np.asarray([camera.params[0],  # fx
                                     camera.params[1],  # fy
                                     camera.params[2],  # cx
                                     camera.params[3],  # cy
                                     0])  # k1

            else:
                raise NotImplementedError('Model {} is not implemented!'.format(camera.model_name))

            camera_params = Camera(id=camera.camera_id,
                                   model=camera.model_name,
                                   width=camera.width,
                                   height=camera.height,
                                   params=params)

            self.cameras.update({camera_id: camera_params})

    def __read_images(self):
        """
        Load infos about images from either image.bin or image.txt file and saves it into an Image object which contains
        information about camera_id, camera extrinsics, image_name, triangulated points (3D), keypoints (2D), etc.

        @return:
        """
        if self._image_path.suffix == '.txt':
            self.images = read_images_text(self._image_path)
        elif self._image_path.suffix == '.bin':
            self.images = read_images_binary(self._image_path)
        else:
            raise FileNotFoundError('Wrong extension found, {}'.format(self._camera_path.suffix))

    def __read_sparse_model(self):
        """
        Read sparse points from either points3D.bin or points3D.txt file. Every point is saved as an Point3D object
        containing information about error, image_ids (from which image can this point be seen?), points2D-idx
        (which keypoint idx is the observation of this triangulated point), rgb value and xyz position.

        @return:
        """
        if self._points3D_path.suffix == '.txt':
            self.sparse = read_points3D_text(self._points3D_path)
        elif self._points3D_path.suffix == '.bin':
            self.sparse = read_points3d_binary(self._points3D_path)
        else:
            raise FileNotFoundError('Wrong extension found, {}'.format(self._camera_path.suffix))

    def __read_depth_structure(self):
        """
        Loads the path for both depth map types ('geometric and photometric') of the reconstruction project.

        @return:
        """
        self.depth_path_geometric = []
        self.depth_path_photometric = []

        for depth_path in list(self._depth_image_path.glob('*.bin')):
            if 'geometric' in depth_path.__str__():
                self.depth_path_geometric.append(depth_path.__str__())
            elif 'photometric' in depth_path.__str__():
                self.depth_path_photometric.append(depth_path.__str__())
            else:
                raise ValueError('Unkown depth image type: {}'.format(depth_path))

    def __read_dense_model(self):
        """
        Load dense point cloud from path.

        @return:
        """
        self.dense = o3d.io.read_point_cloud(self._fused_path.__str__())


    def write(self, data):
        pass
