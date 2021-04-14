import logging

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

logger = logging.getLogger(__name__)


def detect_and_predict_mask(image, faceNet, maskNet, threshold=0.5):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image=image,
        scalefactor=1.0,
        size=(224, 224),
        mean=(104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    logger.info(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces, locs, preds = [], [], []

    # loop for the detections
    for face in range(0, detections.shape[2]):

        # extract the confidence (i.e probability) associated with the detection
        confidence = detections[0, 0, face, 2]

        # filter out weak detections by ensuring the confidence is greather than the minimum confidence
        if confidence > threshold:
            # compute the (x, y) pixel coordinates of the bounding box for the object
            box = detections[0, 0, face, 3:7] * np.array([w, h, w, h])


def load_models():
    def rchop(s, suffix):
        if suffix and s.endswith(suffix):
            return s[:-len(suffix)]
        return s

    # load our serialized face detector model from disk
    DIR = rchop(f"{os.path.dirname(__file__)}", '/facemask_detector')
    prototxtPath = rf"{DIR}/trained_models/deploy.prototxt"
    weightsPath = rf"{DIR}/trained_models/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model(rf"{DIR}/trained_models/mask_detector.model")
    return faceNet, maskNet


def main():
    # initialize the videostream
    # vs = VideoStream(src=0).start()

    # grab the frame from the threaded video stream and resize it to have a
    # max width of 400 pixels
    # image = vs.read()
    # image = imutils.resize(image, width=400)
    load_models()


if __name__ == "__main__":
    main()
