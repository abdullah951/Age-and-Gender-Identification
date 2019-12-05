import base64

from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer
import json

import cv2 as cv
import cv2
import math
import time
import argparse


class FileView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def getFaceBox(self, net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > conf_threshold:

                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
        return frameOpencvDnn, bboxes

    def post(self, request, *args, **kwargs):
        global image_data
        global encoded_string
        file_serializer = FileSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            # print(file_serializer.data.get('remark'))
            # info = json.loads(json.loads(file_serializer.data))
            # result = json.loads(str(file_serializer.data))
            print(file_serializer.data.get('file'))
            print(file_serializer.data.get('remark'))
            if file_serializer.data.get('remark') == 'image':
                faceProto = "file_app/opencv_face_detector.pbtxt"
                faceModel = "file_app/opencv_face_detector_uint8.pb"

                ageProto = "file_app/age_deploy.prototxt"
                ageModel = "file_app/age_net.caffemodel"

                genderProto = "file_app/gender_deploy.prototxt"
                genderModel = "file_app/gender_net.caffemodel"

                MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
                ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
                genderList = ['Male', 'Female']

                # Load network
                ageNet = cv.dnn.readNet(ageModel, ageProto)
                genderNet = cv.dnn.readNet(genderModel, genderProto)
                faceNet = cv.dnn.readNet(faceModel, faceProto)

                # Open a video file or an image file or a camera stream
                cap = cv.VideoCapture(r'C:\Users\hahaha\PycharmProjects\Fyp' + file_serializer.data.get('file'))
                padding = 20
                facedetected = False
                m = {}
                while cv.waitKey(1) < 0:
                    # Read frame
                    t = time.time()
                    hasFrame, frame = cap.read()
                    if not hasFrame:
                        cv.waitKey()
                        break

                    frameFace, bboxes = self.getFaceBox(faceNet, frame)
                    if not bboxes:
                        print("No face Detected, Checking next frame")
                        continue

                    for bbox in bboxes:
                        # print(bbox)
                        facedetected=True
                        print(facedetected)
                        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                        genderNet.setInput(blob)
                        genderPreds = genderNet.forward()
                        gender = genderList[genderPreds[0].argmax()]
                        # print("Gender Output : {}".format(genderPreds))
                        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

                        ageNet.setInput(blob)
                        agePreds = ageNet.forward()
                        age = ageList[agePreds[0].argmax()]
                        print("Age Output : {}".format(agePreds))
                        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

                        label = "{},{}".format(gender, age)
                        cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.95,
                                   (0, 255, 255),
                                   2, cv.LINE_AA)
                        # cv.imshow("Age Gender Demo", frameFace)
                        cv.imwrite("age-gender-out.jpg", frameFace)

                    print("time : {:.3f}".format(time.time() - t))
                    print(file_serializer.data)
                    print(facedetected)

                    if facedetected:

                        with open("age-gender-out.jpg", "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read())
                            n=12

                            #m = {'type': 'image', 'base64': [encoded_string[i:i + n] for i in range(0, len(encoded_string), n)]}
                            m = {'type': 'image',
                                 'base64': encoded_string}
                            #print(len(m.get('base64')))
                    else:
                        encoded_string = None
                        m = {'type': 'image',
                             'base64': 'None'}
                        print(m.get('base64'))
                return Response(encoded_string)
            elif file_serializer.data.get('remark') == 'video':
                faceProto = "file_app/opencv_face_detector.pbtxt"
                faceModel = "file_app/opencv_face_detector_uint8.pb"

                ageProto = "file_app/age_deploy.prototxt"
                ageModel = "file_app/age_net.caffemodel"

                genderProto = "file_app/gender_deploy.prototxt"
                genderModel = "file_app/gender_net.caffemodel"

                MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
                ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
                genderList = ['Male', 'Female']

                # Load network
                ageNet = cv.dnn.readNet(ageModel, ageProto)
                genderNet = cv.dnn.readNet(genderModel, genderProto)
                faceNet = cv.dnn.readNet(faceModel, faceProto)

                # Open a video file or an image file or a camera stream
                cap = cv.VideoCapture(r'C:\Users\hahaha\PycharmProjects\Fyp' + file_serializer.data.get('file'))
                padding = 20
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter("output.avi", fourcc, 20.0,
                                     (frame_width, frame_height))
                encoded_string = ""
                facedetected = False
                while cv.waitKey(1) < 0:
                    # Read frame
                    t = time.time()
                    hasFrame, frame = cap.read()
                    if not hasFrame:
                        cv.waitKey()
                        break

                    frameFace, bboxes = self.getFaceBox(faceNet, frame)
                    if not bboxes:
                        print("No face Detected, Checking next frame")
                        out.write(frame)
                        continue

                    for bbox in bboxes:
                        # print(bbox)
                        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                        genderNet.setInput(blob)
                        genderPreds = genderNet.forward()
                        gender = genderList[genderPreds[0].argmax()]
                        # print("Gender Output : {}".format(genderPreds))
                        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

                        ageNet.setInput(blob)
                        agePreds = ageNet.forward()
                        age = ageList[agePreds[0].argmax()]
                        print("Age Output : {}".format(agePreds))
                        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
                        label = ""
                        if agePreds[0].max() > 0.7:
                            label = "{},{},{},{}".format(gender, genderPreds[0].max(), age, agePreds[0].max())
                        cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                                   (0, 255, 255),
                                   2, cv.LINE_AA)
                        out.write(frameFace)
                        # cv.imshow("Age Gender Demo", frameFace)
                        # cv.imwrite("age-gender-out.jpg", frameFace)
                    print("time : {:.3f}".format(time.time() - t))
                    # print(file_serializer.data)

                    with open("output.avi", "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read())
                return Response(image_file)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
