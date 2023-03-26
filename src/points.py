"""
Created By:
    Baher Kher Bek
"""

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import numpy
import cv2
import matplotlib.pyplot as plt
class Points:
    def __init__(self, x, y=[], img=None, ScaleFactor=1):
        self.x = x
        self.y = y
        self.Colors = None
        self.Labels = None
        self.imgshape = None
        self.ClusteredImage = None

    def ScaleImg(self, img, ScaleFactor):
        pass

    def LineFit(self, x, y):
        x = numpy.array(x).reshape(-1, 1)
        y = numpy.array(y).reshape(-1, 1)
        regr = LinearRegression()
        regr.fit(x, y)

        return regr

    def ScaleMatrix(self):
        pass

    def KMEANS(self, x, numClusters):
        if self.x.ndim == 3:
            self.imgshape = x.shape
            x = x.reshape((x.shape[0] * x.shape[1], 3))
        kmeans = KMeans(n_clusters=numClusters)
        kmeans.fit(x)
        self.Colors = kmeans.cluster_centers_
        self.Labels = kmeans.labels_

        return kmeans


    def VisualizeColors(self, colors, labels):
        centers = numpy.uint8(colors)
        self.ClusteredImage = centers[labels.flatten()]
        self.ClusteredImage = self.ClusteredImage.reshape(self.imgshape)
        return self.ClusteredImage



if __name__ == '__main__':
    pass
