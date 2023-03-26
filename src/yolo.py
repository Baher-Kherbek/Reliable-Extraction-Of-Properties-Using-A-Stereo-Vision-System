"""
Created By:
    Baher Kher Bek
"""

import sys
import sklearn
import numpy

sys.path.insert(1, '/home/baher/Desktop/projects/Graduation/src/yolov5-master')
sys.path.insert(1, '/home/baher/Desktop/projects/Graduation/weights')


from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import xyxy2xywh, check_img_size, non_max_suppression, Profile, scale_boxes
from utils.augmentations import letterbox
from utils.plots import Annotator, colors, save_one_box
import numpy as np
import torch
import cv2
from sklearn.linear_model import LinearRegression

class YOLO():
    def __init__(
            self,
            imgsz=320,
            half=False,
            device='',
            conf=0.25,
            iou=0.45,
            classes=None,
            agnostic=False,
            max_det=1000,
            line_thickness=3
    ):
        self.weights = '/home/baher/Desktop/projects/Graduation/weights/best2.pt'
        self.data = 'cus.yaml'
        self.imgsz = (imgsz, imgsz)
        self.half = half
        self.model = None
        self.device = select_device(device)
        self.conf_thres = conf
        self.iou_thres = iou
        self.classes = classes
        self.agnostic_nms = agnostic
        self.max_det = max_det
        self.line_thickness = line_thickness
        self.xyxy = None
        self.XYXY = []
        self.XYWH = []
        self.LambdaX = 5 / 200
        self.LambdaY = 5 / 200
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data, fp16=self.half)
        self.imgsz = check_img_size(self.imgsz)
        self.pred = None
        self.im0 = None
        self.im = None
        self.img = None
        self.IsolateImage = None
        self.AnnotatedImage = None
        self.Prediction = None
        self.quadrants = None
        self.BoundingBoxIsolation = None

    def getPrediction(self, img):
        frame = numpy.copy(img)
        self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else 1, 3, *self.imgsz))
        dt =(Profile(), Profile(), Profile())

        im0 = frame
        im = letterbox(im0, self.imgsz, stride=self.model.stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        with dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        #Inference
        with dt[1]:
            pred = self.model(im, augment=False)

        with dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        self.pred = pred
        self.im = im
        self.im0 = im0

        return pred, im, im0

    def getParameters(self, pred, im0):
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        det = pred[0]

        lines = []
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf)
            lines.append(line)
            lines.append(xyxy)
        return lines

    def AnnotateDetections(self, pred, im, im0, hide_labels=False, hide_conf=False, Cords=False, additionalText=''):

        annotator = Annotator(np.ascontiguousarray(im0), line_width=self.line_thickness, example=str(self.model.names))
        det = pred[0]
        self.XYXY = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = None if hide_labels else (
                self.model.names[c] if hide_conf else f'{self.model.names[c]} {100 * conf:.2f}% {additionalText}')
            annotator.box_label(xyxy, label, color=colors(c, True))
            im0 = annotator.result()
            xyxy = list(map(int, xyxy))
            self.XYXY.append(xyxy)

        return im0

    def getDetections(self, pred, im, im0):
        det = pred[0]
        Detections = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            Detections.append([xyxy])

        return Detections

    def IsolateObject(self, img, XYWH):
        isolated = np.copy(img)
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        for xywh in XYWH:
            cv2.grabCut(img, mask, xywh, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            isolated = img * mask2[:, :, np.newaxis]

        return isolated


if __name__ == '__main__':
    img = cv2.imread('/home/baher/Desktop/projects/Graduation/datasets/fruitsandVeggies/train_data/images/train/good_quality_9.jpg')
    weights = '/home/baher/Desktop/projects/Graduation/weights/best2.pt'
    data = 'cus.yaml'
    model = YOLO(weights, data, conf=0.2, imgsz=160)

    pred, im, im0 = model.getPrediction(img)
    Prediction = model.getParameters(pred, im0)
    Annotated = model.AnnotateDetections(pred, im, im0)

    Detections = model.getDetections(pred, im, im0)


    print(Detections)

    xyxy = model.XYXY
    print(xyxy)

    for xy in xyxy:
        Annotated = cv2.circle(Annotated, (xy[0], xy[1]), 10, [0, 0, 255], -1)
        Annotated = cv2.circle(Annotated, (xy[2], xy[3]), 10, [0, 0, 255], -1)

    cv2.imshow('frame', Annotated)
    cv2.waitKey(0)





