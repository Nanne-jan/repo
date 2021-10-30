import unittest

from detector import Detector, Classes
import cv2 as cv
import numpy as np


class SourceTest(unittest.TestCase):

    def setUp(self):
        self.detector = Detector(trainedModel="frozen_inference_graph.pb",
                                 modelConfig="ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
        self.image = "test_picture_1.jpg"
        self.frame = cv.imread(self.image)

        self.classes = Classes()

    def tearDown(self):
        pass

    def test_classes_pick_class(self):
        self.assertEqual(self.classes.pick_class(index=1), "person")

    def test_setup_network(self):
        self.detector.setup_network()
        self.detector.input_frame = self.frame
        self.detector.set_input()
        result = self.detector.forward()
        self.assertIsInstance(result, np.ndarray)

    def test_detect(self):
        result = self.detector.detect(frame=self.frame)
        self.assertIsInstance(result, np.ndarray)

        for detection in result[0, 0, :, :]:
            score = float(detection[2])
            # print(score)
            if score > 0.3:
                print('detected')

    def test_detector(self):
        result = self.detector.detect(frame=self.frame)
        img = self.detector._detector(network_output=result, certainty=30.0)
        cv.imshow("image", img)
        cv.waitKey(5000)

    def test_movie_detect(self):
        cap = cv.VideoCapture(1, cv.CAP_DSHOW)
        # cap = cv.VideoCapture("rtsp://admin:XPCWU3ZbiERSNav1v3Xz@192.168.2.9:554/live/video/profile1")
        while True:
            success, frame = cap.read()
            found, img = self.detector.detect(frame=frame, what_to_detect=["person", "potted plant"])
            print(found)
            if found:
                print("found person")
            cv.imshow("video", img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    def test_threshold(self):
        for cnt in range(4):
            self.detector.detect_count_add(1)
            print(self.detector.detect_count)
        print(self.detector.detect_count)
