# ShipRSImageNet_devkit

[![python](https://img.shields.io/badge/Python-3.x-ff69b4.svg)](https://github.com/luyanger1799/Amazing-Semantic-Segmentation.git)[![Apache](https://img.shields.io/badge/Apache-2.0-blue.svg)](https://github.com/luyanger1799/Amazing-Semantic-Segmentation.git)

The ShipRSImageNet Development kit is based on DOTA Development kit, and provides some useful function of ShipRSImageNet.
## Functions

The code is useful for ShipRSImageNet. The code provide the following function
<ul>
    <li>
        Load and image, and show the bounding box on it.
    </li>
    <li>
        Covert VOC format label to COCO format label.
    </li>
</ul>


### What is ShipRSImageNet?
ShipRSIageNet is a large-scale fine-grained dataset for ship detection in high-resolution optical remote sensing images. The dataset contains 3,435 images from various sensors, satellite platforms, locations, and seasons. Each image is around 930×930 pixels and contains ships with different scales, orientations, and aspect ratios. The images are annotated by experts in satellite image interpretation, categorized into 50 object categories. The fully annotated ShipRSImageNet contains 17,573 ship instances. 


### Installation
1. install swig
```
    sudo apt-get install swig
```
2. create the c++ extension for python
```
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
```

### Usage
1. Reading and visualizing data, you can use ShipRSImageNet.py
2. Converting the VOC format label to COCO format label, you can use ShipRSImageNet2COCO.py

