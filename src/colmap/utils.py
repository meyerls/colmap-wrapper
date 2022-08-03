#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
# ...

# Libs
import numpy as np
import open3d as o3d


# Own modules
# ...


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def convert_colmap_extrinsics(frame):
    Rwc = frame.Rwc()
    twc = frame.twc()

    M = np.eye(4)
    M[:3, :3] = Rwc
    M[:3, 3] = twc

    return Rwc, twc, M


def generate_colmap_sparse_pc(points3D: np.ndarray) -> o3d.pybind.geometry.PointCloud:
    sparse_pc = np.zeros((points3D.__len__(), 3))
    sparse_pc_color = np.zeros((points3D.__len__(), 3))

    for idx, pc_idx in enumerate(points3D.__iter__()):
        sparse_pc[idx] = points3D[pc_idx].xyz
        sparse_pc_color[idx] = points3D[pc_idx].rgb / 255.

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(sparse_pc)
    pc.colors = o3d.utility.Vector3dVector(sparse_pc_color)

    return pc
