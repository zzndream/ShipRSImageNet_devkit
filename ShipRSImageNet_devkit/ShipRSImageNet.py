#The code is used for visulization ShipRSImageNet dataset, inspired from cocoapi and based on DOTA_devkit
#  Licensed under the Simplified BSD License [see bsd.txt]

import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection0
from matplotlib.patches import Polygon, Circle
import numpy as np
from ShipRSImageNet_devkit import ShipRSImageNet_utils as util
from collections import defaultdict
import cv2
import random


def _isArrayLike(obj):
    if type(obj) == str:
        return False
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class ShipRSImageNet:
    def __init__(self, basepath):
        self.basepath = basepath
        self.labelpath = os.path.join(basepath, 'Annotations')
        self.imagepath = os.path.join(basepath, 'JPEGImages')
        self.imgpaths = util.GetFileFromThisRootDir(self.labelpath)
        self.imglist = [util.custombasename(x) for x in self.imgpaths]
        self.catToImgs = defaultdict(list)
        self.ImgToAnns = defaultdict(list)
        self.createIndex()

    def createIndex(self):
        for filename in self.imgpaths:
            objects = util.parse_ShipRSImageNet_poly(filename)
            imgid = util.custombasename(filename)
            self.ImgToAnns[imgid] = objects
            for obj in objects:
                cat = obj['name']
                self.catToImgs[cat].append(imgid)

    def getImgIds(self, catNms=[]):
        """
        :param catNms: category names
        :return: all the image ids contain the categories
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        if len(catNms) == 0:
            return self.imglist
        else:
            imgids = []
            for i, cat in enumerate(catNms):
                if i == 0:
                    imgids = set(self.catToImgs[cat])
                else:
                    imgids &= set(self.catToImgs[cat])
        return list(imgids)

    def loadAnns(self, catNms=[], imgId = None, difficult=None):
        """
        :param catNms: category names
        :param imgId: the img to load anns
        :return: objects
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        objects = self.ImgToAnns[imgId]
        if len(catNms) == 0:
            return objects
        outobjects = [obj for obj in objects if (obj['name'] in catNms)]
        return outobjects
    def showAnns(self, objects, imgId):
        """
        :param catNms: category names
        :param objects: objects to show
        :param imgId: img to show
        :return:
        """
        img = self.loadImgs(imgId)[0]
        plt.imshow(img)
        plt.axis('off')

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        circles = []
        r = 5
        for obj in objects:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = obj['poly']
            polygons.append(Polygon(poly))
            color.append(c)
            point = poly[0]
            circle = Circle((point[0], point[1]), r)
            circles.append(circle)
        p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        p = PatchCollection(circles, facecolors='red')
        ax.add_collection(p)
        plt.show()
    def loadImgs(self, imgids=[]):
        """
        :param imgids: integer ids specifying img
        :return: loaded img objects
        """
        # print('isarralike:', _isArrayLike(imgids))
        imgids = imgids if _isArrayLike(imgids) else [imgids]
        # print('imgids:', imgids)
        imgs = []
        for imgid in imgids:
            filename = os.path.join(self.imagepath, imgid + '.bmp')
            # print('filename:', filename)
            img = cv2.imread(filename)
            imgs.append(img)
        return imgs

if __name__ == '__main__':
    examplesplit = ShipRSImageNet('/home/ssd/dataset/ShipRSImageNet/VOC_Format/')
    imgids = examplesplit.getImgIds(catNms=[])
    num_show = 30
    img = examplesplit.loadImgs(imgids[:num_show])
    imgids_slice = random.sample(imgids,num_show)
    for imgid in imgids_slice:
        anns = examplesplit.loadAnns(imgId=imgid)
        examplesplit.showAnns(anns, imgid)
    print('Down')

