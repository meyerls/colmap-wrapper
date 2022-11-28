#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.


Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

# Built-in/Generic Imports
import struct

# Libs
import numpy as np
import pathlib as path

# Own
from colmap_wrapper.colmap.camera import (CAMERA_MODEL_IDS, Camera, ImageInformation, Point3D)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    # Struct unpack return tuple (https://docs.python.org/3/library/struct.html)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_cameras_binary(a):
    """
    void Reconstruction::WriteCamerasBinary(const std::string& path) const {
      std::ofstream file(path, std::ios::trunc | std::ios::binary);
      CHECK(file.is_open()) << path;

      WriteBinaryLittleEndian<uint64_t>(&file, cameras_.size());

      for (const auto& camera : cameras_) {
        WriteBinaryLittleEndian<camera_t>(&file, camera.first);
        WriteBinaryLittleEndian<int>(&file, camera.second.ModelId());
        WriteBinaryLittleEndian<uint64_t>(&file, camera.second.Width());
        WriteBinaryLittleEndian<uint64_t>(&file, camera.second.Height());
        for (const double param : camera.second.Params()) {
          WriteBinaryLittleEndian<double>(&file, param);
        }
      }
    }

    @param a:
    @return:
    """
    pass
    return


def read_cameras_binary(path_to_model_file):
    """
    Original C++ Code can be found here: https://github.com/colmap/colmap/blob/dev/src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        # First 8 bits contain information about the quantity of different camera models
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras): # camera_line_index
            # Afterwards the 64 bits contain information about a specific camera
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            # The next  NUM_PARAMS * 8 bits contain information about the camera parameters
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_binary(path_to_model_file):
    """
    Original C++ Code can be found here: https://github.com/colmap/colmap/blob/dev/src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        # First 8 bits contain information about the quantity of different registrated camera models
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images): # image_index
            # Image properties: (image_id, qvec, tvec, camera_id)
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            # Normalized rotation quaternion - 4 entries
            qvec = np.array(binary_image_properties[1:5])
            # Translational Part  - 3 entries
            tvec = np.array(binary_image_properties[5:8])
            # Camera ID
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            # Number of 2D image features detected
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            # 2D location of features in image + Feature ID (x,y,id)
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
            # 2D location of features in image
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            # Feature ID
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))

            pt3did_to_kpidx = {}
            for kpidx, ptid in enumerate(point3D_ids.ravel().tolist()):
                if ptid != -1:
                    if ptid in pt3did_to_kpidx:
                        # print("3D point {} already exits in {}, skip".format(
                        # ptid, image_name))
                        continue
                    pt3did_to_kpidx[ptid] = kpidx

            images[image_id] = ImageInformation(image_id=image_id,
                                     qvec=qvec,
                                     tvec=tvec,
                                     camera_id=camera_id,
                                     image_name=image_name,
                                     xys=xys,
                                     point3D_ids=point3D_ids,
                                     point3DiD_to_kpidx=pt3did_to_kpidx)
    return images


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))

                pt3did_to_kpidx = {}
                for kpidx, ptid in enumerate(point3D_ids.ravel().tolist()):
                    if ptid != -1:
                        if ptid in pt3did_to_kpidx:
                            # print("3D point {} already exits in {}, skip".format(
                            # ptid, image_name))
                            continue
                        pt3did_to_kpidx[ptid] = kpidx

                images[image_id] = ImageInformation(image_id=image_id,
                                         qvec=qvec,
                                         tvec=tvec,
                                         camera_id=camera_id,
                                         image_name=image_name,
                                         xys=xys,
                                         point3D_ids=point3D_ids,
                                         point3DiD_to_kpidx=pt3did_to_kpidx)
    return images


def read_points3d_binary(path_to_model_file):
    """
    Original C++ Code can be found here: https://github.com/colmap/colmap/blob/dev/src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        # Number of points in sparse model
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points): # point_line_index
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            # Point ID
            point3D_id = binary_point_line_properties[0]
            # XYZ
            xyz = np.array(binary_point_line_properties[1:4])
            # RGB
            rgb = np.array(binary_point_line_properties[4:7])
            # What kind of error?
            error = np.array(binary_point_line_properties[7])
            # Number of features that observed this 3D Point
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            # Feature ID connected to this point
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length)
            # Image ID connected to this 3D Point
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                point_id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(point_id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


# ToDo: Image reader


def read_reconstruction_data(base_path):
    '''
        Path to colmap folder. It must contain folder 'sparse' with files 'cameras.bin', 'images.bin', 'points3D.bin'.

    Returns cameras, images, points3D

    :param path:
    :return: cameras, images, points3D
    '''
    reco_base_path = path.Path(base_path)
    sparse_base_path = reco_base_path.joinpath('sparse')
    camera_path = sparse_base_path.joinpath('cameras.bin')
    image_path = sparse_base_path.joinpath('images.bin')
    points3D_path = sparse_base_path.joinpath('points3D.bin')

    points3D = read_points3d_binary(points3D_path)
    cameras = read_cameras_binary(camera_path)
    images = read_images_binary(image_path)

    return cameras, images, points3D


def read_cameras_text(path, int_id=True):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                if int_id:
                    camera_id = int(elems[0])
                else:
                    camera_id = elems[0]
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()