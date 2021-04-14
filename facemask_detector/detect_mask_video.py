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

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > threshold:
            # compute the (x, y) pixel coordinates of the bounding box for the object
            box = detections[0, 0, face, 3:7] * np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding locations
    return locs, preds


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

    # load models
    faceNet, maskNet = load_models()

    # initialize the videostream
    vs = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, image = vs.read()

        # grab the frame from the threaded video stream and resize it to have a
        # max width of 400 pixels
        #dsize = (frame.shape[1], 400)
        #image = cv2.resize(frame, dsize)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # When everything done, release the capture
    image.release()
    cv2.destroyAllWindows()
    vs.release()


if __name__ == "__main__":
    main()
