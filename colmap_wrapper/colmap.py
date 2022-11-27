#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

# Built-in/Generic Imports

# Libs

# Own modules

try:
    from colmap_wrapper.utils import *
    from colmap_wrapper.visualization import *
    from colmap_wrapper.bin import *
    from colmap_wrapper.gps import GPSVis
    from colmap_wrapper.colmap_project import COLMAPProject
except ImportError:
    from .gps import GPSVis
    from .utils import *
    from .visualization import *
    from .bin import *
    from .colmap_project import COLMAPProject


class COLMAP():
    def __init__(self, project_path: str,
                 dense_pc='fused.ply',
                 load_images: bool = True,
                 load_depth: bool = False,
                 image_resize: float = 1.,
                 bg_color: np.ndarray = np.asarray([1, 1, 1])):

        self.vis_bg_color = bg_color
        self._project_path: pathlib.Path = path.Path(project_path)

        if '~' in str(self._project_path):
            self._project_path = self._project_path.expanduser()

        self._sparse_base_path = self._project_path.joinpath('sparse')
        if not self._sparse_base_path.exists():
            raise ValueError('Colmap project structure: sparse folder (cameras, images, points3d) can not be found')

        project_structure = {}

        for project_index, sparse_project_path in enumerate(list(self._sparse_base_path.iterdir())):
            project_structure.update({project_index: {"sparse": sparse_project_path}})

        self._dense_base_path = self._project_path.joinpath('dense')
        for project_index, dense_project_path in enumerate(list(self._dense_base_path.iterdir())):
            project_structure[project_index].update({"dense": dense_project_path})

        self.projects = []
        self.model_ids = []
        for project_index in project_structure.keys():
            self.model_ids.append(project_index)

            project = COLMAPProject(project_path=project_structure[project_index],
                                    dense_pc='fused.ply',
                                    load_images=load_images,
                                    load_depth=load_depth,
                                    image_resize=0.4,
                                    bg_color=bg_color)

            self.projects.append(project)

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
        self.projects[model_idx].geometries = []
        self.projects[model_idx].add_colmap_dense2geometrie()
        self.projects[model_idx].add_colmap_sparse2geometrie()
        self.projects[model_idx].add_colmap_frustums2geometrie(frustum_scale=frustum_scale, image_type=image_type)
        self.projects[model_idx].start_visualizer(point_size=point_size)

    def visualization_all(self, frustum_scale: float = 1., point_size: float = 1., image_type: str = 'image', *args):
        """

        @param frustum_scale:
        @param point_size:
        @param image_type: ['image, depth_geo', 'depth_photo']
        """
        image_types = ['image', 'depth_geo', 'depth_photo']

        if image_type not in image_types:
            raise TypeError('image type is {}. Only {} is allowed'.format(image_type, image_types))

        self.geometries = []

        for model_idx in self.model_ids:
            self.projects[model_idx].add_colmap_dense2geometrie()
            self.projects[model_idx].add_colmap_sparse2geometrie()
            self.projects[model_idx].add_colmap_frustums2geometrie(frustum_scale=frustum_scale, image_type=image_type)
            self.geometries.extend(self.projects[model_idx].geometries)

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
    # project = COLMAP(project_path='data/door', load_images=True, load_depth=True, image_resize=0.4)
    # project = COLMAP(project_path='/home/luigi/Dropbox/07_data/CherrySLAM/test_sequences/01_easy/reco/',
    project = COLMAP(project_path='/home/luigi/Dropbox/07_data/For5G/22_11_14/reco',
                     dense_pc='fused.ply',
                     load_images=True,
                     load_depth=False,
                     image_resize=0.4)

    project_0 = project.projects[0]
    project_1 = project.projects[1]

    MODEL_IDX = 0

    camera = project_0.cameras
    images = project_0.images
    sparse = project_0.get_sparse()
    dense = project_0.get_dense()

    # registrate(project.get_dense()[0], project.get_dense()[1])

    # project.visualization(frustum_scale=0.8, image_type='image', model_idx=MODEL_IDX)
    project.visualization_all(frustum_scale=0.8, image_type='image')
