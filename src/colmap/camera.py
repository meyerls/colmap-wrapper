#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
import struct
import collections

# Libs
import cv2
import numpy as np
import open3d as o3d
import pathlib as path
import PIL

# Own modules
try:
    from utils import *
except ModuleNotFoundError:
    from .utils import *

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


def load_image(image_path: str) -> np.ndarray:
    """
    Load Image. This takes almost 50% of the time. Would be nice if it is possible to speed up this process. Any
    ideas?

    :param image_path:
    :return:
    """
    return np.asarray(PIL.Image.open(image_path))


class Image(object):
    def __init__(self, image_id, qvec, tvec, camera_id, name, image_path, xys, point3D_ids, point3DiD_to_kpidx):
        self.id = image_id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.image_path = image_path
        self.xys = xys
        self.point3D_ids = point3D_ids
        self.point3DiD_to_kpidx = point3DiD_to_kpidx

        self.__image = None

        self.downsample = 0.3

    @property
    def image(self):
        if self.__image is None:
            img = load_image(self.image_path)
            self.__image = cv2.resize(img, (0, 0), fx=self.downsample, fy=self.downsample)
            del img

        return self.__image

    @image.setter
    def image(self, image: np.ndarray):
        self.__image = image

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

    def qtvec(self):
        return self.qvec.ravel().tolist() + self.tvec.ravel().tolist()

    def Rwc(self):
        return self.qvec2rotmat().transpose()

    def twc(self):
        return np.dot(-self.qvec2rotmat().transpose(), self.tvec)

    def Rcw(self):
        return self.qvec2rotmat()

    def tcw(self):
        return self.tvec

    def Twc(self):
        Twc = np.eye(4)
        Twc[0:3, 3] = self.twc()
        Twc[0:3, 0:3] = self.Rwc()

        return Twc

    def Tcw(self):
        Tcw = np.eye(4)
        Tcw[0:3, 3] = self.tcw()
        Tcw[0:3, 0:3] = self.Rcw()

        return Tcw


class Intrinsics:
    def __init__(self, camera):
        self._cx = None
        self._cy = None
        self._fx = None
        self._fy = None

        self.camera = camera
        self.load_from_colmap(camera=self.camera)

    def load_from_colmap(self, camera):
        self.fx = camera.params[0]
        self.fy = camera.params[1]
        self.cx = camera.params[2]
        self.cy = camera.params[3]

    @property
    def cx(self):
        return self._cx

    @cx.setter
    def cx(self, cx):
        self._cx = cx

    @property
    def cy(self):
        return self._cy

    @cy.setter
    def cy(self, cy):
        self._cy = cy

    @property
    def fx(self):
        return self._fx

    @fx.setter
    def fx(self, fx):
        self._fx = fx

    @property
    def fy(self):
        return self._fy

    @fy.setter
    def fy(self, fy):
        self._fy = fy

    @property
    def K(self):
        K = np.asarray([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0, 0, 1]])

        return K


if __name__ == '__main__':
    import json
    import pathlib
    import matplotlib.pyplot as plt

    image_id = 1
    qvec = np.array([0.95161614, -0.05707647, -0.3016161, -0.01402605])
    tvec = np.array([0.65905613, 0.10833697, 0.21707027])
    camera_id = 1
    image_name = 'DSC_7772.JPG'
    xys = np.loadtxt('test/xys.txt')
    point3D_ids = np.loadtxt('test/point3D_ids.txt')
    with open('test/pt3did_to_kpidx.json', 'r') as file:
        pt3did_to_kpidx = json.load(file)

    image = Image(image_id=image_id,
                  qvec=qvec,
                  tvec=tvec,
                  camera_id=camera_id,
                  name=image_name,
                  image_path=(pathlib.Path('../../data/door/images').resolve() / image_name).resolve(),
                  xys=xys,
                  point3D_ids=point3D_ids,
                  point3DiD_to_kpidx=pt3did_to_kpidx)

    img = image.image
    plt.imshow(img)
    plt.show()
