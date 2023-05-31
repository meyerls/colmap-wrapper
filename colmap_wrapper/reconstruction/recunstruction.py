"""
Copyright (c) 2023 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
from pathlib import Path
import json
from copy import copy
import os
from dataclasses import dataclass

# Libs
import pycolmap


# Own modules
# ...


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


@dataclass
class COLMAPReconstruction(object):
    # Images input folder
    image_path: str

    # Output folder for reconstruction
    output_path: str

    # Max window size for patch match stereo
    patch_match_max_image_size: int = 4000

    # Max window size for stereo fusion
    stereo_fusion_max_image_size: int = 4000

    def __init__(self, images, output, feature_mathing, patch_match_max_image_size, stereo_fusion_max_image_size):

        self.images_folder = Path(images).expanduser().resolve()
        project_files = (p.resolve() for p in Path(self.images_folder).glob("**/*") if
                         p.suffix in {".jpg", ".jpeg", ".png", ".ppm"})
        self.projects = {}
        # Define default folders and multiple projects
        if len(list(project_files)) == 0:
            for project_idx, folder in enumerate(self.images_folder.glob("*")):
                reco_path = Path(output).expanduser().resolve() / str(project_idx)
                option_path = reco_path / "options"

                self.projects.update(
                    {
                        project_idx:
                            {
                                'images': Path(folder),
                                'output': reco_path,
                                'sparse': reco_path / "sparse",
                                'mvs': reco_path / "dense",
                                'database': reco_path / "database.db",
                                'option': option_path
                            }
                    })

                reco_path.mkdir(parents=True, exist_ok=True)
                option_path.mkdir(exist_ok=True)
        else:
            reco_path = Path(output).expanduser().resolve()
            option_path = reco_path / "options"

            self.projects.update(
                {
                    0:
                        {
                            'images': Path(images).expanduser().resolve(),
                            'output': reco_path,
                            'sparse': reco_path / "sparse",
                            'mvs': reco_path / "dense",
                            'database': reco_path / "database.db",
                            'option': option_path
                        }
                })
            reco_path.mkdir(parents=True, exist_ok=True)
            option_path.mkdir(exist_ok=True)

        # Feature options
        self.sift_extraction_options = pycolmap.SiftExtractionOptions()
        # self.sift_extraction_options.max_image_size = -1
        # self.sift_extraction_options.max_num_features = 14000
        self.sift_matching_options = pycolmap.SiftMatchingOptions()

        # Feature Matching
        self.feature_mathing = feature_mathing
        if feature_mathing == "exhaustive":
            self.matching_options = pycolmap.ExhaustiveMatchingOptions()
        elif feature_mathing == "spatial":
            self.matching_options = pycolmap.SpatialMatchingOptions()
            self.matching_options.ignore_z = False

        # SfM Options
        self.incremental_mapping_options = pycolmap.IncrementalMapperOptions()

        # MVS Options
        self.patch_match_options = pycolmap.PatchMatchOptions()
        self.patch_match_options.window_radius = 8
        self.patch_match_options.num_iterations = 7
        self.patch_match_options.max_image_size = patch_match_max_image_size
        self.stereo_fusion_options = pycolmap.StereoFusionOptions()
        self.stereo_fusion_options.max_image_size = stereo_fusion_max_image_size

        self.camera = pycolmap.Camera(
            model='SIMPLE_PINHOLE',
            width=8192,
            height=5460,
            params=[],
        )

        self.camera_dslr = pycolmap.Camera(
            model='OPENCV',
            width=6000,
            height=4000,
            params=[4518.9, 4511.7, 3032.2, 2020.9, -0.1623, 0.0902, 0, 0],
        )

        self.camera_mode = pycolmap.CameraMode('SINGLE')

        if pycolmap.has_cuda:
            print("COLMAP Reconstruction uses CUDA.")
        else:
            print("COLMAP Reconstruction uses NO CUDA! Using CPU instead.")

    @staticmethod
    def save_options(output_file, options):
        jsonable = is_jsonable(options)

        if jsonable:
            with open(output_file, 'w') as file:
                json.dump(options, file)
        else:
            raise TypeError("Option not JSON serializable: {}".format(options))

    @staticmethod
    def load_options(input_file):
        options = None

        if input_file.exists():

            with open(input_file, 'r') as file:
                options = json.load(file)
        else:
            options = False
        return options

    def check_for_existing(self, input_file, current_options):
        loaded_options = self.load_options(input_file=input_file)

        if loaded_options:
            if loaded_options == current_options:
                return False
            else:
                return True
        return True

    def extract_features(self):
        for project_idx in self.projects.keys():
            options = copy(self.sift_extraction_options.todict())
            # Make jasonable
            options['normalization'] = options['normalization'].name

            ret = self.check_for_existing(self.projects[project_idx]['option'] / "feature_extraction_options.json",
                                          current_options=options)

            if ret:
                # Options do not exist or existing options differ from current
                pycolmap.extract_features(self.projects[project_idx]['database'],
                                          self.projects[project_idx]['images'],
                                          camera_mode=self.camera_mode,
                                          device=pycolmap.Device.auto,
                                          sift_options=self.sift_extraction_options)

                self.save_options(self.projects[project_idx]['option'] / "feature_extraction_options.json",
                                  options)
            else:
                print('Feature extraction skipped. Options have not changed. Access database data...')
                # Load existing sift features from database
                # import_images(self.images_folder, self.database_path, self.camera_mode)
                # image_ids = get_image_ids(self.database_path)
                # import_features(image_ids, self.database_path, features)

    def feature_matcher(self):
        for project_idx in self.projects.keys():
            options = copy({**self.matching_options.todict(), **self.sift_matching_options.todict()})
            ret = self.check_for_existing(self.projects[project_idx]['option'] / "feature_matching_options.json",
                                          current_options=options)

            if ret:
                if self.feature_mathing == 'exhaustive':
                    matcher = pycolmap.match_exhaustive
                elif self.feature_mathing == 'spatial':
                    matcher = pycolmap.match_spatial

                # Options do not exist or existing options differ from current
                matcher(database_path=self.projects[project_idx]['database'],
                        device=pycolmap.Device.auto,
                        sift_options=self.sift_matching_options,
                        matching_options=self.matching_options)
                self.save_options(self.projects[project_idx]['option'] / "feature_matching_options.json", options)

            else:
                print('Exhaustive feature matching skipped. Options have not changed. Access database data...')

                # Load existing matches from database
                # import_matches()

    def incremental_sfm(self):
        for project_idx in self.projects.keys():
            options = copy(self.incremental_mapping_options.todict())
            options["image_names"] = []
            ret = self.check_for_existing(self.projects[project_idx]['option'] / "incremental_sfm_options.json",
                                          current_options=options)
            if ret:
                maps = pycolmap.incremental_mapping(database_path=self.projects[project_idx]['database'],
                                                    image_path=self.images_folder,
                                                    output_path=self.projects[project_idx]['sparse'],
                                                    options=self.incremental_mapping_options)
                self.save_options(self.projects[project_idx]['option'] / "incremental_sfm_options.json", options)
                maps[0].write(self.projects[project_idx]['sparse'])

            else:
                print('Incremental mapping skipped. Options have not changed. Access sparse file data...')

    def undistort_images(self):
        for project_idx in self.projects.keys():
            if not (self.projects[project_idx]['mvs'] / "images").exists():
                pycolmap.undistort_images(self.projects[project_idx]['mvs'],
                                          self.projects[project_idx]['sparse'],
                                          self.projects[project_idx]['images'])
            else:
                print('Images already undistorted! Skipping...')

    def patch_match_stereo(self):
        for project_idx in self.projects.keys():
            options = copy(self.patch_match_options.todict())
            ret = self.check_for_existing(self.projects[project_idx]['option'] / "patch_match_stereo_options.json",
                                          current_options=options)

            if ret:
                pycolmap.patch_match_stereo(self.projects[project_idx]['mvs'], options=self.patch_match_options)
                self.save_options(self.projects[project_idx]['option'] / "patch_match_stereo_options.json", options)
            else:
                print('Patch match stereo images already processed. Skipping...')

    def stereo_fusion(self):
        for project_idx in self.projects.keys():
            self.stereo_fusion_options.num_threads = min(16, os.cpu_count())
            options = copy(self.stereo_fusion_options.todict())
            options['bounding_box'] = str([list(array) for array in options['bounding_box']])
            is_jsonable(options)
            ret = self.check_for_existing(self.projects[project_idx]['option'] / "stereo_fusion_options.json",
                                          current_options=options)

            dense_model_path = self.projects[project_idx]['mvs'] / "fused.ply"
            if ret:
                pycolmap.stereo_fusion(output_path=dense_model_path,
                                       workspace_path=self.projects[project_idx]['mvs'],
                                       workspace_format='COLMAP',
                                       pmvs_option_name='option-all',
                                       input_type='geometric', options=self.stereo_fusion_options)
                self.save_options(self.projects[project_idx]['option'] / "stereo_fusion_options.json", options)
            else:
                print('Stereo images already fused to model. Skipping...')

    def run(self):
        # Sparse Reconstruction / SFM - Structure from Motion
        self.extract_features()
        self.feature_matcher()
        self.incremental_sfm()

        # Dense Reconstruction / MVS - Multi View Stereo
        self.undistort_images()
        self.patch_match_stereo()
        self.stereo_fusion()


if __name__ == '__main__':
    from colmap_wrapper.data.download import *
    from colmap_wrapper.visualization import ColmapVisualization
    from colmap_wrapper.dataloader import COLMAPLoader

    downloader = Dataset()
    downloader.download_bunny_images_dataset()

    reconstruction = COLMAPReconstruction(images=downloader.file_path,
                                          output='./test_reco',
                                          feature_mathing='exhaustive',
                                          patch_match_max_image_size=400,
                                          stereo_fusion_max_image_size=400)
    reconstruction.run()

    project = COLMAPLoader(project_path=downloader.file_path)

    project_vs = ColmapVisualization(colmap=project, image_resize=0.4)
    project_vs.visualization(frustum_scale=0.8, image_type='image')
