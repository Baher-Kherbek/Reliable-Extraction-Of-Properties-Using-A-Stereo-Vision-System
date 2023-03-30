"""
Author:
	Baher Kher Bek
"""

import cv2.aruco as aruco
import cv2
import numpy
disparity = []
alpha = []

i = 0
class FRAME:
    def __init__(self):
        self.ids = None
        self.width = None
        self.height = None
        self.Referencelength = None
        self.bbxs = None
        self.frame = None
        self.DetectedArucos = None
        self.Referencelength = 9.5
        self.ArucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.ArucoParams = aruco.DetectorParameters()
        self.ArucoDetector = aruco.ArucoDetector(self.ArucoDict, self.ArucoParams)

    def Process(self, img):
        self.frame = img
        self.height, self.width = self.frame.shape[:2]

        #Detect Aruco
        self.bbxs, self.ids = self.DetectAruco(img)

        #Draw Aruco
        if self.ids is not None and len(self.ids) >= 2:
            self.DetectedArucos = self.DrawAruco(img, self.bbxs)
            self.count = self.BoxCount(img, self.bbxs)
        else:
            self.DetectedArucos = None

    def DetectAruco(self, img):
        bbxs, ids, _ = self.ArucoDetector.detectMarkers(img)
        return bbxs, ids

    def DrawAruco(self, img, bbxs, color=(0,0,255), thickness=3):
        frame = numpy.copy(img)
        length = 0
        for aruco in bbxs:
            tl = list(map(int, aruco[0][0]))
            tr = list(map(int, aruco[0][1]))
            br = list(map(int, aruco[0][2]))
            bl = list(map(int, aruco[0][3]))

            self.bbx = [tl[0], tl[1], br[0], br[1]]

            frame = cv2.line(frame, tl, tr, color[::-1], thickness)
            frame = cv2.line(frame, tr, br, color, thickness)
            frame = cv2.line(frame, br, bl, color, thickness)
            frame = cv2.line(frame, bl, tl, color, thickness)

        return frame

    def ConvertBBX(self, bbxs):
        BoundingBoxes = []
        for aruco in bbxs:
            tl = list(map(int, aruco[0][0]))
            br = list(map(int, aruco[0][2]))
            BoundingBoxes.append([tl[0], tl[1], br[0], br[1]])

        return BoundingBoxes


    def CurveFit(self):
        pass

    def getVariables(self, bbxs):
        if self.ids is not None and len(self.ids) >= 2:
            length = 0
            for aruco in bbxs:
                tlx = self.NormalizeWidth(aruco[0][0][0])
                tly = self.NormalizeWidth(aruco[0][1][0])
                length += abs(tly - tlx)

            x1 = self.NormalizeWidth(bbxs[0][0][0][0])
            x2 = self.NormalizeWidth(bbxs[1][0][0][0])

            disparity = abs(x1 - x2)
            alpha = self.Referencelength / length
            return (disparity, alpha)

        return None

    def NormalizeWidth(self, x):
        if x >= self.width//2:
            return int(x - self.width//2)
        return x

    def BoxCount(self, frame, bbxs):
        BoundingBoxes = self.ConvertBBX(bbxs)
        count = []
        for box in BoundingBoxes:
            xmin, ymin, xmax, ymax = box
            self.Area = (xmax - xmin) * (ymax - ymin)
            slice = frame[ymin: ymax, xmin: xmax]
            #slice = cv2.cvtColor(slice, cv2.COLOR_BGR2GRAY)
            #slice = self.StrechHisto(slice)
            if len(frame.shape) == 3:
                count.append(numpy.sum(slice != 0) // 3)
            else:
                count.append(numpy.sum(slice != 0))

        return count

cap = cv2.VideoCapture(0)

frame = FRAME()

while True:
    Frame = cap.read()[1]
    frame.Process(Frame)
    cv2.imshow('Raw', frame.frame if frame.DetectedArucos is None else frame.DetectedArucos)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('s'):
        i += 1
        point = frame.getVariables(frame.bbxs)
        if point is not None:
            disparity.append(point[0])
            alpha.append(point[1])
            numpy.save('/home/baher/Desktop/projects/Graduation/main/CurveFitting/data/disparities.npy', disparity)
            numpy.save('/home/baher/Desktop/projects/Graduation/main/CurveFitting/data/alpha.npy', alpha)
            cv2.imwrite(f'/home/baher/Desktop/projects/Graduation/main/CurveFitting/Images/Image{i}.jpg', frame.frame)
        print(f'Disparities: {disparity} \n Alpha: {alpha} \n Count: {frame.count} \n Area: {frame.Area}' )


