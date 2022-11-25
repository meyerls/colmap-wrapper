# COLMAP Wrapper

<a href="https://img.shields.io/pypi/pyversions/colmap-wrapper"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/colmap-wrapper"></a>
<a href="https://github.com/meyerls/colmap-wrapper/actions"><img alt="GitHub Workflow Status" src="https://img.shields.io/github/workflow/status/meyerls/colmap-wrapper/Python%20package"></a>
<a href="https://github.com/meyerls/colmap_wrapper/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/github/license/meyerls/colmap-wrapper"></a>

## About 

Colmap wrapper is a library to work with colmap projects. The purpose is the simplification to read e.g. rgb images, depth
images, camera poses, sparse point clouds etc. Additionally a visualization for a colmap project is provided.

<p align="center">
    <img width="40%" src="img/img_1.png">
    <img width="50%" src="img/img_2.png">
</p>

## Installation

Make sure that you have a Python version >=3.7 installed.

This repository is tested on Python 3.6+ and can currently only be installed
from [PyPi](https://pypi.org/project/colmap-wrapper/).

 ````bash
pip install colmap-wrapper
 ````

## Usage

```python
from colmap_wrapper.colmap import COLMAP

project = COLMAP(project_path="[PATH2COLMAP_PROJECT]", load_images=True, load_depth=True, image_resize=0.4)

# Acess camera, images and sparse + dense point cloud
camera = project.cameras
images = project.images
sparse = project.get_sparse()
dense = project.get_dense()

project.visualization(frustum_scale=0.2, image_type='image')
```

## References

GPS visualization is done by: https://github.com/tisljaricleo/GPS-visualization-Python 
A guide to visualize gps data can be found here: https://towardsdatascience.com/simple-gps-data-visualization-using-python-and-open-street-maps-50f992e9b676 