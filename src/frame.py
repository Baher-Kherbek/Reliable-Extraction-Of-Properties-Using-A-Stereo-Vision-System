"""
Created By:
    Baher Kher Bek
"""

import cv2
import numpy as np
import numpy
from yolo import YOLO
from points import Points
import matplotlib
import matplotlib.pyplot as plt


class FRAME(YOLO, Points):
    def __init__(self, ScalePercent=100):
        super().__init__(conf=0.2)
        self.quadrants = None
        self.ScalePercent = ScalePercent
        self.ThresholedImage = None
        self.ThresholedCrisp = None

    def ProcessFrame(self, img):
        #Scale Image
        width = int(img.shape[1] * self.ScalePercent / 100)
        height = int(img.shape[0] * self.ScalePercent / 100)

        self.img = cv2.resize(img, (width, height))
        self.img = img

        # Classification and Detection
        self.Classify(img)

        #Isolation
        if len(self.Prediction):
            self.XYWH = self.XYXYtoXYWH(self.XYXY)
            self.IsolateObject(img, self.XYWH)
            self.IsolateBoundingBox(self.img)
        else:
            self.IsolatedImage = img


        #Measurements
        self.getMeasurement(img)
        self.AnnotateMeasurements(self.IsolatedImage)

        #Cluster Colors
        self.x = self.IsolatedImage
        kmeans = self.KMEANS(self.x, numClusters=2)
        self.ClusteredImgage = self.VisualizeColors(self.Colors, self.Labels)

        #Threshold
        self.CrispThreshold(self.IsolatedImage)

    def Classify(self, img):
        self.pred, self.im, self.im0 = self.getPrediction(img)
        self.Prediction = self.getParameters(self.pred, self.im0)
        self.AnnotatedClassification = self.AnnotateDetections(self.pred, self.im, self.im0)

    def CrispThreshold(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, self.ThresholedImage = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
        kernel = numpy.ones((15, 15), numpy.uint8)
        self.ThresholedCrisp = cv2.morphologyEx(self.ThresholedImage, cv2.MORPH_OPEN, kernel)

    def XYXYtoXYWH(self, BBX):
        Points = numpy.copy(BBX)

        for idx, point in enumerate(Points):
            width = point[2] - point[0]
            height = point[3] - point[1]
            XYWH = [point[0], point[1], width, height]
            Points[idx] = XYWH

        return Points

    def IsolateObject(self, img, XYWH):
        isolated = np.copy(img)
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        for xywh in XYWH:
            cv2.grabCut(img, mask, xywh, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            self.IsolatedImage = img * mask2[:, :, np.newaxis]

    def getQuadrants(self, img, XYWH):
        gray = numpy.copy(img)
        w, h = gray.shape[:2]
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        array = numpy.zeros((w, h), np.uint8)
        array[:, XYWH[0][0] + XYWH[0][2] // 2] = gray[:, XYWH[0][0] + XYWH[0][2] // 2]
        array[XYWH[0][1] + XYWH[0][3] // 2, :] = gray[XYWH[0][1] + XYWH[0][3] // 2, :]

        index = numpy.where(array != 0)
        index = list(zip(index[1], index[0]))

        srt = numpy.sort(index, axis=1)
        top = srt[0]
        bottom = srt[-1]

        srt = numpy.sort(index, axis=0)

        left = (srt[0][0], srt[0][1] + XYWH[0][3] // 2)
        right = (srt[-1][0], srt[-1][1] - XYWH[0][3] // 2)

        self.quadrants = [top, bottom, left, right]

    def getMeasurement(self, img):
        self.getQuadrants(self.IsolatedImage, self.XYWH)
        top, bottom, left, right = self.quadrants
        self.height = abs(top[1] - bottom[1]) * self.LambdaY
        self.width = abs(right[0] - left[0]) * self.LambdaX

    def AnnotateMeasurements(self, image, radius=2, color=[255,0,255], gap=5):
        img = np.copy(image)
        top, bottom, left, right = self.quadrants

        #height
        points = self.drawLine(top, bottom, gap=gap)

        for point in points:
            img = cv2.circle(img, point, radius, color, -1)

        #Width
        points = self.drawLine(left, right, gap=gap)

        for point in points:
            img = cv2.circle(img, point, radius, color, -1)

        self.AnnotatedDimensions = img

        # Put Text
        self.AnnotatedDimensions = cv2.putText(
            img=self.AnnotatedDimensions,
            text=f'Height : {self.height}cm x Width: {self.width}cm',
            org=(left[0], top[1]-20) if top[1] >= 50 else ((left[0], bottom[1]+20)),
            fontFace=cv2.FONT_ITALIC,
            fontScale=0.5,
            color=(125, 246, 55),
            thickness=1
        )

    def drawLine(self, pt1, pt2, gap):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
        pts = []
        for i in np.arange(dist//100, dist-dist//100, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            p = (x, y)
            pts.append(p)

        return pts

    def ShowPlots(self, rows, cols, *add):
        matplotlib.use('TkAgg')
        fig = plt.figure(figsize=(9, 13))
        cols, rows = (cols, rows)
        ax = []
        ax.append(fig.add_subplot(rows, cols, 1))
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        ax[-1].set_title("Raw Frame")

        ax.append(fig.add_subplot(rows, cols, 2))
        plt.imshow(cv2.cvtColor(self.AnnotatedClassification, cv2.COLOR_BGR2RGB))
        ax[-1].set_title("Detection and Classification")

        ax.append(fig.add_subplot(rows, cols, 3))
        plt.imshow(cv2.cvtColor(self.BoundingBoxIsolation, cv2.COLOR_BGR2RGB))
        ax[-1].set_title("Bounding Box Isolation")

        ax.append(fig.add_subplot(rows, cols, 4))
        plt.imshow(cv2.cvtColor(self.IsolatedImage, cv2.COLOR_BGR2RGB))
        ax[-1].set_title("Isolation using\nGrabcut Algorithm")

        ax.append(fig.add_subplot(rows, cols, 5))
        plt.imshow(cv2.cvtColor(self.AnnotatedDimensions, cv2.COLOR_BGR2RGB))
        ax[-1].set_title("Dimension Extraction")

        ax.append(fig.add_subplot(rows, cols, 6))
        plt.imshow(cv2.cvtColor(self.ClusteredImage, cv2.COLOR_BGR2RGB))
        ax[-1].set_title("K-Means Clustering \n for color extraction")

        ax.append(fig.add_subplot(rows, cols, 7))
        plt.imshow(self.ThresholedImage, cmap='gray')
        ax[-1].set_title("Threshold")

        ax.append(fig.add_subplot(rows, cols, 8))
        plt.imshow(self.ThresholedCrisp, cmap='gray')
        ax[-1].set_title("Morphology on resulting Threshold")
        plt.savefig('Result.png')
        plt.show()

    def IsolateBoundingBox(self, img):
        # Bounding Box Isolation
        xyxy = self.XYWH[0]
        xyxy = list(map(int, xyxy))
        self.BoundingBoxIsolation = numpy.zeros(img.shape, np.uint8)
        self.BoundingBoxIsolation[xyxy[1]: xyxy[1] + xyxy[2], xyxy[0]: xyxy[0] + xyxy[3], :] = img[xyxy[1]: xyxy[1] + xyxy[2], xyxy[0]: xyxy[0] + xyxy[3], :]


if __name__ == '__main__':
    pass
