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
            print(f"detections: {detections[0, 0, face, 3:7]}")
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


def pose_detection(image, poseNet, threshold=0.2):
    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

    LEG_PARTS = {"RKnee": 9, "RAnkle": 10, "LKnee": 12, "LAnkle": 13}
    LEG_POSE_PAIRS = [["RKnee", "RAnkle"], ["LKnee", "LAnkle"]]

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image=image,
        scalefactor=1.0,
        size=(224, 224),
        mean=(104.0, 177.0, 123.0),
        swapRB=True,
        crop=False
    )

    # pass the blob through the network and obtain the face detections
    poseNet.setInput(blob)
    detections = poseNet.forward()
    detections = detections[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    logger.info(detections.shape)

    assert (len(BODY_PARTS) == detections.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = detections[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (w * point[0]) / detections.shape[3]
        y = (h * point[1]) / detections.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > threshold else None)

    for pair in LEG_POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = LEG_PARTS[partFrom]
        idTo = LEG_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(image, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(image, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(image, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    t, _ = poseNet.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    cv2.putText(image, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    lk = points[LEG_PARTS["LKnee"]]
    la = points[LEG_PARTS["LAnkle"]]
    rk = points[LEG_PARTS["RKnee"]]
    ra = points[LEG_PARTS["RAnkle"]]

    if lk and la and rk and ra:
        print(f"lk {lk}, la {la}, rk {rk}, ra {ra}")
        image = bounding_box_around_feet(image, lk, la, rk, ra)
    return image


def bounding_box_around_feet(image, left_knee, left_ankle, right_knee, right_ankle):
    def extrapolate(p1, p2, scale=1.0):
        (x1, y1) = p1
        (x2, y2) = p2
        x3 = x1 + (x2 - x1) * scale
        y3 = y1 + (y2 - y1) * scale
        return x3, y3

    def leg_length(p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2
        dist = ((((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 0.5)
        return dist

    def get_box_coord(point, w, h):
        x, y = point[0], point[1]

        start_x = (x - round(w / 2))
        end_x = (x + round(w / 2))
        start_y = (y - round(h / 2))
        end_y = (y + round(h / 2))
        return (round(start_x), round(start_y)), (round(end_x), round(end_y))

    left_foot = extrapolate(left_knee, left_ankle, 1.5)
    right_foot = extrapolate(right_knee, right_ankle, 1.5)

    left_box_size = leg_length(left_knee, left_ankle) * 0.6
    right_box_size = leg_length(right_knee, right_ankle) * 0.6

    l_start, l_end = get_box_coord(left_foot, left_box_size, left_box_size)
    r_start, r_end = get_box_coord(right_foot, right_box_size, right_box_size)

    color = (0, 0, 255)

    cv2.putText(image, "left foot", (l_start[0], l_start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(image, l_start, l_end, color, 2)

    cv2.putText(image, "right foot", (r_start[0], r_start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(image, r_start, r_end, color, 2)
    return image


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
    poseNet = cv2.dnn.readNetFromTensorflow(rf"{DIR}/trained_models/pose_detection.pb")
    return faceNet, maskNet, poseNet


def main():
    # load models
    faceNet, maskNet, poseNet = load_models()

    # initialize the videostream
    vs = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, image = vs.read()

        # grab the frame from the threaded video stream and resize it to have a
        # max width of 400 pixels
        # dsize = (frame.shape[1], 400)
        # image = cv2.resize(frame, dsize)

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

        # pose detection
        image = pose_detection(image, poseNet)

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
