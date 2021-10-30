import unittest

from source import Source
import cv2 as cv
import numpy as np


class SourceTest(unittest.TestCase):

    def setUp(self):
        self.video = "nj.avi"
        self.source = Source(source=self.video)
        _frm = self.source.get_frame
        self._frame_to_test = next(_frm)

        self.vc = cv.VideoCapture(self.video)
        success, self.test_frame = self.vc.read()

        _height = int(self.vc.get(3))
        _width = int(self.vc.get(4))
        self._size = _width, _height

    def tearDown(self):
        self.vc.release()

    def test_stream(self):
        self.source.stream = "test-string"
        self.assertEqual(self.source.stream, "test-string")

    def test_get_frame(self):
        _test_frm = self.source.get_frame
        _frame = next(_test_frm)
        print(type(_frame))
        print("shape", _frame.shape)
        self.assertIsInstance(_frame, np.ndarray)

    def test_source_size(self):
        self.assertEqual(self.source.get_size, self._size)


if __name__ == '__main__':
    unittest.main()
