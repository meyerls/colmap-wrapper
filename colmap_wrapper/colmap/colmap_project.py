#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

# Built-in/Generic Imports
import warnings
from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing import cpu_count
from enum import Enum
from pathlib import Path

# Libs
import pycolmap
import numpy as np
import open3d as o3d
import exiftool

# Own modules
from colmap_wrapper.colmap import (Camera, Intrinsics, read_array, read_images_text, read_points3D_text,
                                   read_points3d_binary, read_images_binary, generate_colmap_sparse_pc)


class LoadElement(Enum):
    PATHS_AND_ATTRIBUTES = 0
    CAMERAS = 1
    IMAGES = 2
    SPARSE_MODEL = 3
    DENSE_MODEL = 4
    DEPTH_STRUCTURE = 5
    EXIF_DATA = 6
    DEPTH_IMAGE = 7
    IMAGE_INFO = 8

class LoadElementStatus:
    def __init__(self, element: LoadElement, project, finished=False, idx=None, current_id=None, max_id=None):
        self.element = element
        self.project = project
        self.finished = finished
        
        if idx != None:
            self.idx = idx
            self.current_id = -1
            self.max_id = 1
            
        if max_id != None or current_id != None:
            self.current_id = current_id
            self.max_id = max_id
    
    def isStarted(self):
        return not self.finished
    
    def isFinished(self):
        return self.finished
    
    def getElement(self):
        return self.element

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


class COLMAPProject(PhotogrammetrySoftware):
    def __init__(self, project_path: [dict, str],
                 dense_pc: str = 'fused.ply',
                 load_depth: bool = False,
                 image_resize: float = 1.,
                 bg_color: np.ndarray = np.asarray([1, 1, 1]),
                 exif_read=False,
                 output_status_function=None):
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
        
        self.output_status_function = output_status_function
        if output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.PATHS_AND_ATTRIBUTES, project=self, finished=False))
        
        # Flag to read exif data (takes long for large image sets)
        self.exif_read = exif_read

        # Search and Set Paths
        if isinstance(project_path, str):
            self._project_path: Path = Path(project_path)

            if '~' in str(self._project_path):
                self._project_path: Path = self._project_path.expanduser()

            self._sparse_base_path = self._project_path.joinpath('sparse')
            if not self._sparse_base_path.exists():
                raise ValueError('Colmap project structure: sparse folder (cameras, images, points3d) can not be found')
            if self._sparse_base_path.joinpath('0').exists():
                self._sparse_base_path: Path = self._sparse_base_path.joinpath('0')

            self._dense_base_path = self._project_path.joinpath('dense')
            if self._dense_base_path.joinpath('0').exists():
                self._dense_base_path: Path = self._dense_base_path.joinpath('0')

            if not self._dense_base_path.exists():
                self._dense_base_path: Path = self._project_path
        elif isinstance(project_path, dict):
            self._project_path: Path = project_path['project_path']
            if not project_path['dense'].exists():
                self._dense_base_path: Path = self._project_path
            else:
                self._dense_base_path: Path = project_path['dense']
            self._sparse_base_path: Path = project_path['sparse']
        else:
            raise ValueError("{}".format(self._project_path))

        # Loads undistorted images
        self._src_image_path: Path = self._dense_base_path.joinpath('images')
        self._fused_path: Path = self._dense_base_path.joinpath(dense_pc)
        self._stereo_path: Path = self._dense_base_path.joinpath('stereo')
        self._depth_image_path: Path = self._stereo_path.joinpath('depth_maps')
        self._normal_image_path: Path = self._stereo_path.joinpath('normal_maps')

        self.__project_ini_path: Path = self._sparse_base_path.joinpath('project.ini')

        files: list = []
        types: tuple = ('*.txt', '*.bin')
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

        self.load_depth: bool = load_depth
        self.image_resize: float = image_resize
        self.vis_bg_color: np.ndarray = bg_color
        self.project_ini = self.__read_project_init_file()
        
        if output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.PATHS_AND_ATTRIBUTES, project=self, finished=True))
        
        self.read()
        
        if output_status_function: # Maybe not reset?
            self.output_status_function = None

    def read(self):
        """
        Start reading all necessary information

        @return:
        """
        
        n_cores = cpu_count()
        executor = ThreadPoolExecutor(max_workers=n_cores)
        
        futures: list = []
        
        futures.append(executor.submit(self.__read_cameras))
        futures.append(executor.submit(self.__read_images))
        futures.append(executor.submit(self.__read_sparse_model))
        futures.append(executor.submit(self.__read_dense_model))
        futures.append(executor.submit(self.__read_depth_structure))
        
        wait(futures)
        futures.clear()
        
        futures.append(executor.submit(self.__add_infos, executor))
        futures.append(executor.submit(self.__read_exif_data))
        
        wait(futures)
        executor.shutdown(wait=True)

    def __read_project_init_file(self):
        if self.__project_ini_path.exists():
            PROJECT_CLASS = 'Basic'
            project_ini = {PROJECT_CLASS: {}}
            with open(self.__project_ini_path.__str__(), 'r') as file:
                for line in file:
                    elements = line.split('=')
                    if len(elements) == 1:
                        PROJECT_CLASS = elements[0].strip('\n')
                        project_ini.update({PROJECT_CLASS: {}})
                        continue
                    project_ini[PROJECT_CLASS].update({elements[0]: elements[1].strip('\n')})
            return project_ini
        else:
            return {}

    def __read_exif_data(self):
        if self.output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.EXIF_DATA, project=self, finished=False))
        
        if self.exif_read:
            if self.__project_ini_path.exists():
                try:
                    for image_idx in self.images.keys():
                        self.images[image_idx].original_filename: Path = Path(self.project_ini['Basic']['image_path']) / \
                                                                         self.images[
                                                                             image_idx].name
                        with exiftool.ExifToolHelper() as et:
                            metadata = et.get_metadata(self.images[image_idx].original_filename.__str__())
                        self.images[image_idx].exifdata = metadata[0]
                except exiftool.exceptions.ExifToolExecuteError as error:
                    # traceback.print_exc()
                    warnings.warn("Exif Data could not be read.")
        
        if self.output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.EXIF_DATA, project=self, finished=True))

    def __add_infos(self, executor: ThreadPoolExecutor = None):
        """
        Loads rgb image and depth images from path and adds it to the Image object.

        @warning: this might exceed your storage! Think about adjusting the scaling parameter.

        @return:
        """
                
        self.max_depth_scaler = 0
        self.max_depth_scaler_photometric = 0
        
        current_image = 0
        count_images = len(self.images)
        
        for image_idx in self.images.keys():
            def run(image_idx=image_idx, current_image=current_image, count_images=count_images):
                self.images[image_idx].path = self._src_image_path / self.images[image_idx].name
    
                if self.load_depth:
                    if self.output_status_function:
                        self.output_status_function(LoadElementStatus(element=LoadElement.DEPTH_IMAGE, project=self, finished=False, idx=image_idx, current_id=current_image, max_id=count_images))
                    
                    
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
    
                    min_depth, max_depth = np.percentile(self.images[image_idx].depth_image_photometric, [5, 95])
    
                    if max_depth > self.max_depth_scaler_photometric:
                        self.max_depth_scaler_photometric = max_depth
                    
                    
                    if self.output_status_function:
                        self.output_status_function(LoadElementStatus(element=LoadElement.DEPTH_IMAGE, project=self, finished=True, idx=image_idx, current_id=current_image, max_id=count_images))
                    
                else:
                    self.images[image_idx].depth_image_geometric = None
                    self.images[image_idx].depth_path_photometric = None
                # self.images[image_idx].normal_image = self.__read_depth_images
                
                
                if self.output_status_function:
                    self.output_status_function(LoadElementStatus(element=LoadElement.IMAGE_INFO, project=self, finished=False, idx=image_idx, current_id=current_image, max_id=count_images))
                
                
                self.images[image_idx].intrinsics = Intrinsics(camera=self.cameras[self.images[image_idx].camera_id])
    
                # Fixing Strange Error when cy is negative
                if self.images[image_idx].intrinsics.cx < 0:
                    pass
    
                if self.images[image_idx].intrinsics.cy < 0:
                    pass
                
                
                if self.output_status_function:
                    self.output_status_function(LoadElementStatus(element=LoadElement.IMAGE_INFO, project=self, finished=True, idx=image_idx, current_id=current_image, max_id=count_images))
            
            
            current_image += 1
            if executor != None:
                executor.submit(run)
            else:
                run()
            
    def __read_cameras(self):
        """
        Load camera model from file. Currently only Simple Radial and 'Pinhole' are supported. If the camera settings
        are identical for all images only one camera is provided. Otherwise, every image has its own camera model.

        @return:
        """
        
        if self.output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.CAMERAS, project=self, finished=False))
        
        
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
                raise NotImplementedError('Model {} is not implemented!'.format(camera.model_name))

            camera_params = Camera(id=camera.camera_id,
                                   model=camera.model_name,
                                   width=camera.width,
                                   height=camera.height,
                                   params=params)

            self.cameras.update({camera_id: camera_params})
        
        
        if self.output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.CAMERAS, project=self, finished=True))

    def __read_images(self):
        """
        Load infos about images from either image.bin or image.txt file and saves it into an Image object which contains
        information about camera_id, camera extrinsics, image_name, triangulated points (3D), keypoints (2D), etc.

        @return:
        """
        
        if self.output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.IMAGES, project=self, finished=False))
        
        
        if self._image_path.suffix == '.txt':
            self.images = read_images_text(self._image_path)
        elif self._image_path.suffix == '.bin':
            self.images = read_images_binary(self._image_path)
        else:
            raise FileNotFoundError('Wrong extension found, {}'.format(self._camera_path.suffix))
        
        
        if self.output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.IMAGES, project=self, finished=True))

    def __read_sparse_model(self):
        """
        Read sparse points from either points3D.bin or points3D.txt file. Every point is saved as an Point3D object
        containing information about error, image_ids (from which image can this point be seen?), points2D-idx
        (which keypoint idx is the observation of this triangulated point), rgb value and xyz position.

        @return:
        """
        if self.output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.SPARSE_MODEL, project=self, finished=False))
        
        if self._points3D_path.suffix == '.txt':
            self.sparse = read_points3D_text(self._points3D_path)
        elif self._points3D_path.suffix == '.bin':
            self.sparse = read_points3d_binary(self._points3D_path)
        else:
            raise FileNotFoundError('Wrong extension found, {}'.format(self._camera_path.suffix))
        
        if self.output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.SPARSE_MODEL, project=self, finished=True))

    def __read_depth_structure(self):
        """
        Loads the path for both depth map types ('geometric and photometric') of the reconstruction project.

        @return:
        """
        if self.output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.DEPTH_STRUCTURE, project=self, finished=False))
        
        self.depth_path_geometric = []
        self.depth_path_photometric = []

        for depth_path in list(self._depth_image_path.glob('*.bin')):
            if 'geometric' in depth_path.__str__():
                self.depth_path_geometric.append(depth_path.__str__())
            elif 'photometric' in depth_path.__str__():
                self.depth_path_photometric.append(depth_path.__str__())
            else:
                raise ValueError('Unkown depth image type: {}'.format(path))
        
        if self.output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.DEPTH_STRUCTURE, project=self, finished=True))

    def __read_dense_model(self):
        """
        Load dense point cloud from path.

        @return:
        """
        if self.output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.DENSE_MODEL, project=self, finished=False))
        
        self.dense = o3d.io.read_point_cloud(self._fused_path.__str__())
        
        if self.output_status_function:
            self.output_status_function(LoadElementStatus(element=LoadElement.DENSE_MODEL, project=self, finished=True))


if __name__ == '__main__':
    from colmap_wrapper.data.download import Dataset
    from colmap_wrapper.visualization import ColmapVisualization

    downloader = Dataset()
    downloader.download_bunny_dataset()

    project = COLMAPProject(project_path=downloader.file_path,
                            load_depth=True,
                            image_resize=0.4)

    camera = project.cameras
    images = project.images
    sparse = project.get_sparse()
    dense = project.get_dense()

    project_vs = ColmapVisualization(colmap=project)
    project_vs.visualization(frustum_scale=0.8, image_type='image')
