#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

# Built-in/Generic Imports
import os
import copy

# Libs
import pycolmap

# Own modules

try:
    from colmap_wrapper.utils import *
    from colmap_wrapper.visualization import *
    from colmap_wrapper.bin import *
    from colmap_wrapper.gps import GPSVis
except ImportError:
    from .gps import GPSVis
    from .utils import *
    from .visualization import *
    from .bin import *


class PhotogrammetrySoftware(object):
    def __init__(self, project_path):
        self._project_path = None

        self.sparse = None
        self.dense = None

        self.geometries = []

    def __read_images(self):
        return NotImplementedError

    def get_sparse(self):
        sparse_list = []
        for model_idx in self.sparse:
            sparse_model = generate_colmap_sparse_pc(self.sparse[model_idx])
            sparse_list.append(sparse_model)
        return sparse_list

    def show_sparse(self):
        o3d.visualization.draw_geometries([self.get_sparse()])

    def get_dense(self):
        dense_list = []
        for model_idx in self.dense:
            dense_list.append(self.dense[model_idx])
        return dense_list

    def show_dense(self):
        o3d.visualization.draw_geometries([self.get_dense()])


class COLMAP(PhotogrammetrySoftware):
    def __init__(self, project_path: str,
                 dense_pc: str = 'fused.ply',
                 load_images: bool = True,
                 load_depth: bool = False,
                 image_resize: float = 1.,
                 bg_color: np.ndarray = np.asarray([1, 1, 1])):
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
        @param load_images: flag to load images.
        @param load_depth: flag to load depth images.
        @param image_resize: float to scale images if image size is to large for RAM storage
        @param bg_color: background color for visualization
        """

        PhotogrammetrySoftware.__init__(self, project_path=project_path)

        self._project_path: pathlib.Path = path.Path(project_path)

        if '~' in str(self._project_path):
            self._project_path = self._project_path.expanduser()

        self._sparse_base_path = self._project_path.joinpath('sparse')
        if not self._sparse_base_path.exists():
            raise ValueError('Colmap project structure: sparse folder (cameras, images, points3d) can not be found')

        # Read different models
        if self._sparse_base_path.joinpath('0').exists():
            self._sparse_base_path = list(self._sparse_base_path.iterdir())
        else:
            self._sparse_base_path = [self._sparse_base_path]

        self._dense_base_path = self._project_path.joinpath('dense')
        if self._dense_base_path.joinpath('0').exists():
            self._dense_base_path = list(self._dense_base_path.iterdir())
        elif not self._dense_base_path.exists():
            self._dense_base_path = self._project_path
        else:
            self._dense_base_path = [self._dense_base_path]

        self._project_ini_path = [i.joinpath('project.ini') for i in self._sparse_base_path]

        # Loads undistorted images
        self._src_image_path = [i.joinpath('images') for i in self._dense_base_path]
        self._fused_path = [i.joinpath(dense_pc) for i in self._dense_base_path]
        self._stereo_path = [i.joinpath('stereo') for i in self._dense_base_path]
        self._depth_image_path = [i.joinpath('depth_maps') for i in self._stereo_path]
        self._normal_image_path = [i.joinpath('normal_maps') for i in self._stereo_path]

        files = {}
        types = ('*.txt', '*.bin')
        for i, path_ in enumerate(self._sparse_base_path):
            files.update({i: []})
            for t in types:
                files[i].extend(path_.glob(t))

        self._camera_path = {}
        self._image_path = {}
        self._points3D_path = {}
        for key in files:
            self._camera_path.update({key: []})
            self._image_path.update({key: []})
            self._points3D_path.update({key: []})
            for file_path in files[key]:
                if 'cameras' in file_path.name:
                    self._camera_path[key] = file_path
                elif 'images' in file_path.name:
                    self._image_path[key] = file_path
                elif 'points3D' in file_path.name:
                    self._points3D_path[key] = file_path
                else:
                    raise ValueError('Unkown file in sparse folder')

        self.load_images = load_images
        self.load_depth = load_depth
        self.image_resize = image_resize
        self.vis_bg_color = bg_color

        self.model_ids = list(range(len(self._sparse_base_path)))

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
        self.__read_project_file()
        self.__add_infos()

    def __read_project_file(self):
        fp = open(self._project_ini_path[0].__str__(), 'r')
        Lines = fp.readlines()

        default_key = "DEFAULT"
        self.project_ini = {default_key: {}}
        # Strips the newline character
        for line in Lines:
            line = line.strip()
            if '=' in line:
                argument, value = line.split('=')
                self.project_ini[default_key].update({argument: value})
            else:
                default_key = line.strip('[]')
                self.project_ini.update({default_key: {}})

    def __add_infos(self):
        """
        Loads rgb image and depth images from path and adds it to the Image object.

        @warning: this might exceed your storage! Think about adjusting the scaling parameter.

        @return:
        """
        for model_idx in self.images.keys():
            self.max_depth_scaler = 0
            self.max_depth_scaler_photometric = 0
            for image_idx in self.images[model_idx].keys():
                self.images[model_idx][image_idx].path = self._src_image_path[model_idx] / self.images[model_idx][
                    image_idx].name

                img_exf = load_image_meta_data(image_path=os.path.join(self.project_ini["DEFAULT"]['image_path'],
                                                                       self.images[model_idx][image_idx].path.name))
                self.images[model_idx][image_idx].exifdata = img_exf

                if self.load_images:
                    image = load_image(self.images[model_idx][image_idx].path)
                    if self.image_resize != 1.:
                        image = cv2.resize(image, (0, 0), fx=self.image_resize, fy=self.image_resize)

                    self.images[model_idx][image_idx].image = image
                else:
                    self.images[model_idx][image_idx].image = None

                if self.load_depth:
                    self.images[model_idx][image_idx].depth_image_geometric = read_array(
                        path=next((p for p in self.depth_path_geometric[model_idx] if
                                   self.images[model_idx][image_idx].name in p), None))

                    # print(self.images[image_idx].name)
                    # print(next((p for p in self.depth_path_geometric if self.images[image_idx].name in p), None))
                    # print('\n')

                    min_depth, max_depth = np.percentile(self.images[model_idx][image_idx].depth_image_geometric,
                                                         [5, 95])

                    if max_depth > self.max_depth_scaler:
                        self.max_depth_scaler = max_depth

                    self.images[image_idx].depth_image_photometric = read_array(
                        path=next((p for p in self.depth_path_photometric[model_idx] if
                                   self.images[model_idx][image_idx].name in p), None))

                    min_depth, max_depth = np.percentile(self.images[model_idx][image_idx].depth_image_photometric,
                                                         [5, 95])

                    if max_depth > self.max_depth_scaler_photometric:
                        self.max_depth_scaler_photometric = max_depth

                else:
                    self.images[model_idx][image_idx].depth_image_geometric = None
                    self.images[model_idx][image_idx].depth_path_photometric = None

                self.images[model_idx][image_idx].intrinsics = Intrinsics(
                    camera=self.cameras[model_idx][self.images[model_idx][image_idx].camera_id])

                # Fixing Strange Error when cy is negative
                if self.images[model_idx][image_idx].intrinsics.cx < 0:
                    pass

                if self.images[model_idx][image_idx].intrinsics.cy < 0:
                    pass

    def __read_cameras(self):
        """
        Load camera model from file. Currently only Simple Radial and 'Pinhole' are supported. If the camera settings
        are identical for all images only one camera is provided. Otherwise, every image has its own camera model.

        @return:
        """
        self.cameras = {}
        for i, sparse_path in enumerate(self._sparse_base_path):
            reconstruction = pycolmap.Reconstruction(sparse_path)
            self.cameras.update({i: {}})
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

                self.cameras[i].update({camera_id: camera_params})

    def __read_images(self):
        """
        Load infos about images from either image.bin or image.txt file and saves it into an Image object which contains
        information about camera_id, camera extrinsics, image_name, triangulated points (3D), keypoints (2D), etc.

        @return:
        """
        self.images = {}
        for key in self._image_path:
            self.images.update({key: []})
            if self._image_path[key].suffix == '.txt':
                self.images[key] = read_images_text(self._image_path[key])
            elif self._image_path[key].suffix == '.bin':
                self.images[key] = read_images_binary(self._image_path[key])
            else:
                raise FileNotFoundError('Wrong extension found, {}'.format(self._camera_path[i].suffix))

    def __read_sparse_model(self):
        """
        Read sparse points from either points3D.bin or points3D.txt file. Every point is saved as an Point3D object
        containing information about error, image_ids (from which image can this point be seen?), points2D-idx
        (which keypoint idx is the observation of this triangulated point), rgb value and xyz position.

        @return:
        """
        self.sparse = {}
        for key in self._points3D_path:
            self.sparse.update({key: []})
            if self._points3D_path[key].suffix == '.txt':
                self.sparse[key] = read_points3D_text(self._points3D_path[key])
            elif self._points3D_path[key].suffix == '.bin':
                self.sparse[key] = read_points3d_binary(self._points3D_path[key])
            else:
                raise FileNotFoundError('Wrong extension found, {}'.format(self._camera_path[key].suffix))

    def __read_depth_structure(self):
        """
        Loads the path for both depth map types ('geometric and photometric') of the reconstruction project.

        @return:
        """
        self.depth_path_geometric = {}
        self.depth_path_photometric = {}

        for i, path_ in enumerate(self._depth_image_path):
            self.depth_path_geometric.update({i: []})
            self.depth_path_photometric.update({i: []})
            for depth_path in list(self._depth_image_path[i].glob('*.bin')):
                if 'geometric' in depth_path.__str__():
                    self.depth_path_geometric[i].append(depth_path.__str__())
                elif 'photometric' in depth_path.__str__():
                    self.depth_path_photometric[i].append(depth_path.__str__())
                else:
                    raise ValueError('Unkown depth image type: {}'.format(path))

    def __read_dense_model(self):
        """
        Load dense point cloud from path.

        @return:
        """
        self.dense = {}
        for key in self._points3D_path:
            self.dense.update({key: o3d.io.read_point_cloud(self._fused_path[key].__str__())})

    def add_colmap_dense2geometrie(self, model_idx):
        if np.asarray(self.get_dense()[model_idx].points).shape[0] == 0:
            return False

        self.geometries.append(self.get_dense()[model_idx])

        return True

    def add_colmap_sparse2geometrie(self, model_idx):
        if np.asarray(self.get_sparse()[model_idx].points).shape[0] == 0:
            return False

        self.geometries.append(self.get_sparse()[model_idx])
        return True

    def add_colmap_frustums2geometrie(self, frustum_scale: float = 1., image_type: str = 'image', model_idx=0):
        """

        @param image_type:
        @type frustum_scale: object
        """
        geometries = []
        for image_idx in self.images[model_idx].keys():

            if image_type == 'image':
                image = self.images[model_idx][image_idx].image
            elif image_type == 'depth_geo':
                image = self.images[model_idx][image_idx].depth_image_geometric
                min_depth, max_depth = np.percentile(
                    image, [5, 95])
                image[image < min_depth] = min_depth
                image[image > max_depth] = max_depth
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = (image / self.max_depth_scaler * 255).astype(np.uint8)
                image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
            elif image_type == 'depth_photo':
                image = self.images[model_idx][image_idx].depth_image_photometric
                min_depth, max_depth = np.percentile(
                    image, [5, 95])
                image[image < min_depth] = min_depth
                image[image > max_depth] = max_depth
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            line_set, sphere, mesh = draw_camera_viewport(extrinsics=self.images[model_idx][image_idx].extrinsics,
                                                          intrinsics=self.images[model_idx][image_idx].intrinsics.K,
                                                          image=image,
                                                          scale=frustum_scale)

            geometries.append(mesh)
            geometries.append(line_set)
            geometries.extend(sphere)

        self.geometries.extend(geometries)

    def visualization(self, frustum_scale: float = 1., point_size: float = 1., image_type: str = 'image', model_idx=0,
                      *args):
        """

        @param frustum_scale:
        @param point_size:
        @param image_type: ['image, depth_geo', 'depth_photo']
        """
        image_types = ['image', 'depth_geo', 'depth_photo']

        if image_type not in image_types:
            raise TypeError('image type is {}. Only {} is allowed'.format(image_type, image_types))

        self.add_colmap_dense2geometrie(model_idx)
        self.add_colmap_sparse2geometrie(model_idx)
        self.add_colmap_frustums2geometrie(frustum_scale=frustum_scale, image_type=image_type, model_idx=model_idx)
        self.start_visualizer(point_size=point_size)

    def visualization_all(self, frustum_scale: float = 1., point_size: float = 1., image_type: str = 'image', *args):
        """

        @param frustum_scale:
        @param point_size:
        @param image_type: ['image, depth_geo', 'depth_photo']
        """
        image_types = ['image', 'depth_geo', 'depth_photo']

        if image_type not in image_types:
            raise TypeError('image type is {}. Only {} is allowed'.format(image_type, image_types))

        for model_idx in self.model_ids:
            self.add_colmap_dense2geometrie(model_idx)
            self.add_colmap_sparse2geometrie(model_idx)
            self.add_colmap_frustums2geometrie(frustum_scale=frustum_scale, image_type=image_type, model_idx=model_idx)
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

    def write(self, data):
        pass

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def registrate(source, target):
    threshold = 0.02
    trans_init = np.eye(4)
    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    draw_registration_result(source, target, reg_p2l.transformation)


if __name__ == '__main__':
    # project = COLMAP(project_path='data/door', load_images=True, load_depth=True, image_resize=0.4)
    # project = COLMAP(project_path='/home/luigi/Dropbox/07_data/CherrySLAM/test_sequences/01_easy/reco/',
    project = COLMAP(project_path='/home/luigi/Documents/reco/22_11_14/reco',
                     dense_pc='fused.ply',
                     load_images=True,
                     load_depth=False,
                     image_resize=0.4)

    MODEL_IDX = 0

    camera = project.cameras[MODEL_IDX]
    images = project.images[MODEL_IDX]
    sparse = project.get_sparse()[MODEL_IDX]
    dense = project.get_dense()[MODEL_IDX]

    # registrate(project.get_dense()[0], project.get_dense()[1])

    # project.visualization(frustum_scale=0.8, image_type='image', model_idx=MODEL_IDX)
    project.visualization_all(frustum_scale=0.8, image_type='image')
