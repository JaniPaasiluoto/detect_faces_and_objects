from dotenv import load_dotenv
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
import datetime
import numpy as np
import imutils
import dlib
import time
import cv2
import os

load_dotenv()

print("[INFO] loading models...")
faceXml = "models/face_detection/face-detection-retail-0005/FP32/face-detection-retail-0005.xml"
faceBin = "models/face_detection/face-detection-retail-0005/FP32/face-detection-retail-0005.bin"
ageGenderXml = "models/face_detection/age_gender/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml"
ageGenderBin = "models/face_detection/age_gender/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin"
objectXml = "models/person_car_bike/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.xml"
objectBin = "models/person_car_bike/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.bin"
ageProtoPath = "models/face_detection/age_gender/deploy_age.prototxt"
ageModelPath = "models/face_detection/age_gender/age_net.caffemodel"

net = cv2.dnn.readNet(objectXml, objectBin)
genderNet = cv2.dnn.readNet(ageGenderXml, ageGenderBin)

faceNet = cv2.dnn.readNet(faceXml, faceBin)
ageNet = cv2.dnn.readNetFromCaffe(ageProtoPath, ageModelPath)

CLASSES = ["bicycle", "car", "person"]
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
               "(38-43)", "(48-53)", "(60-100)"]
GENDERS = ["female", "male"]


def detect_and_predict(frame, faceNet, ageNet, genderNet, minConf=0.5):
    # initialize our results list
    results = []
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    faceDetections = faceNet.forward()

    # loop over the detections
    for i in range(0, faceDetections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = faceDetections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > minConf:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = faceDetections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the ROI of the face
            face = frame[startY:endY, startX:endX]
            # ensure the face ROI is sufficiently large
            if face.shape[0] < 30 or face.shape[1] < 30:
                continue

            # construct a blob from *just* the face ROI
            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                             (78.4263377603, 87.7689143744, 114.895847746),
                                             swapRB=False)

            ageGenderBlob = cv2.dnn.blobFromImage(face, size=(62, 62), ddepth=cv2.CV_8U)
            genderNet.setInput(ageGenderBlob)
            detections = genderNet.forwardAndRetrieve(['prob'])
            #
            gender = GENDERS[detections[0][0][0].argmax()]
            # age = detections[1][0][0][0][0][0] * 100

            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
            i = preds[0].argmax()
            age = AGE_BUCKETS[i]
            ageConfidence = preds[0][i]

            # construct a dictionary consisting of both the face
            # bounding box location along with the age prediction,
            # then update our results list
            d = {
                # "loc": (startX, startY, endX, endY),
                "age": (age, ageConfidence),
                "gender": (gender),
            }
            results.append(d)

        return results


def detect_track_count():
    centroidTracker_max_disappeared = int(os.getenv('CENTROID_TRACKER_MAX_DISAPPEARED'))
    centroidTracker_max_distance = int(os.getenv('CENTROID_TRACKER_MAX_DISTANCE'))
    skip_frames = int(os.getenv("SKIP_FRAMES"))

    print("[INFO] warming up camera...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    # initialize the frame dimensions
    H = None
    W = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(centroidTracker_max_disappeared, centroidTracker_max_distance)
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    totalOverall = 0

    while True:
        frame = vs.read()

        if frame is None:
            break

        # resize the frame
        frame = imutils.resize(frame, width=900)
        frame = cv2.flip(frame, 0)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % skip_frames == 0:
            # initialize our new set of object trackers
            trackers = []
            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300),
                                         ddepth=cv2.CV_8U)
            net.setInput(blob, scalefactor=1.0 / 127.5, mean=[127.5,
                                                              127.5, 127.5])
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:
                    # extract the index of the class label
                    idx = int(detections[0, 0, i, 1])
                    if idx > 2:
                        continue
                    # if the class label is not a car or person, ignore it
                    if CLASSES[idx] == "person" or CLASSES[idx] == "car":
                        # compute the (x, y)-coordinates of the bounding box
                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")
                        # construct a dlib rectangle object from the bounding
                        # box coordinates and then start the dlib correlation
                        # tracker
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(bgr, rect)
                        # add the tracker to our list of trackers so we can utilize it during skip frames
                        trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing
        # throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # update the tracker and grab the updated position
                tracker.update(bgr)
                pos = tracker.get_position()
                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))
        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)
            if to is None:
                timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                to = TrackableObject(objectID, centroid)
                if CLASSES[idx] == "car":
                    objectInfo = [("ID", objectID), ("timestamp", timestamp), ("object", CLASSES[idx])]
                    print(objectInfo)

                if CLASSES[idx] == "person":
                    objectInfo = [("ID", objectID), ("timestamp", timestamp), ("object", CLASSES[idx])]
                    results = detect_and_predict(frame, faceNet, ageNet, genderNet)
                    # loop over the results
                    for r in results:
                        objectInfo = [("ID", objectID), ("timestamp", timestamp), ("object", CLASSES[idx]),
                                      ("gender", r["gender"]), ("age", r["age"])]
                    print(objectInfo)

            else:
                y = [c[1] for c in to.centroids]
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    totalOverall += 1
                    to.counted = True

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

        totalFrames += 1


detect_track_count()
