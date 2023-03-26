"""
Created By:
    Baher Kher Bek
"""

from frame import FRAME
import cv2

frame = FRAME()
img = cv2.imread('/home/baher/Desktop/projects/Graduation/5ii.jpg')
frame.ProcessFrame(img)
frame.ShowPlots(1, 8)
