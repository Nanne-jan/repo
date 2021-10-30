import cv2 as cv
import numpy as np


class Classes:
    def __init__(self):
        self._classes = ["background", "person", "bicycle", "car", "motorcycle",
                         "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                         "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                         "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
                         "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
                         "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                         "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
                         "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                         "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
                         "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
                         "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
                         "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        self._colors = np.random.uniform(0, 255, size=(len(self._classes), 3))

    def pick_color(self, **kwargs):
        _color_idx = kwargs.get("index", None)
        if _color_idx is not None:
            _return_color = self._colors[_color_idx]
        else:
            _return_color = ""

        return _return_color

    def pick_class(self, **kwargs) -> str:
        _idx = kwargs.get("index", None)
        if _idx is not None:
            _return_class = self._classes[_idx]
        else:
            _return_class = ""

        return _return_class


class Detector:
    def __init__(self, **kwargs):
        self._classes = Classes()
        self._trained_model: str = kwargs.get("trainedModel", None)
        self._model_config: str = kwargs.get("modelConfig", None)
        self._on_gpu: bool = kwargs.get("runOnGpu", False)
        self._network = None
        self._frame = None
        self._wanted_score: float = 0.3
        self._what_to_detect: list = []
        self._detect_threshold: int = 0
        self._detect_count: int = 0
        self._detector_state: bool = False

    def setup_network(self):
        if self._network is None:
            self._network = cv.dnn.readNetFromTensorflow(self._trained_model, self._model_config)
        else:
            pass  # TODO add logging

    def setup_gpu(self):
        if self._on_gpu and self._network is not None:
            self._network.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self._network.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        else:
            pass  # TODO add logging

    def set_input(self):
        if self._frame is not None:
            if self._network is None:
                self.setup_network()
            self._network.setInput(cv.dnn.blobFromImage(self._frame, size=(300, 300), swapRB=True, crop=False))
        else:
            pass  # TODO add logging

    def forward(self):
        return self._network.forward()

    @property
    def input_frame(self) -> np.ndarray:
        return self._frame

    @input_frame.setter
    def input_frame(self, frame: np.ndarray):
        if isinstance(frame, np.ndarray):
            self._frame = frame
        else:
            pass  # TODO add logging

    @property
    def _frame_size(self):
        if self._frame is not None:
            # size in tuple with shape like (rows, cols)
            return self._frame.shape[0], self._frame.shape[1]

    def _bounded_box(self, _frame: np.ndarray, detection: np.ndarray) -> np.ndarray:
        _lineThickness = 1
        left = detection[3] * self._frame_size[1]  # cols
        top = detection[4] * self._frame_size[0]  # rows
        right = detection[5] * self._frame_size[1]  # cols
        bottom = detection[6] * self._frame_size[0]  # rows
        cv.rectangle(_frame, (int(left), int(top)), (int(right), int(bottom)),
                     (23, 230, 210), _lineThickness)

        return _frame

    def _label(self, detection: np.ndarray) -> str:
        idx = int(detection[1])
        label = "{}".format(self._classes.pick_class(index=idx))

        return label

    @property
    def certainty(self) -> float:
        return self._wanted_score * 100

    @certainty.setter
    def certainty(self, percentage: float):
        if percentage <= 100:
            self._wanted_score = percentage / 100
        else:
            self._wanted_score = 1.0

    def _calc_percentage(self, score: float):
        return score * 100

    @property
    def detect_count(self) -> int:
        return self._detect_count

    # @threshold.setter
    def detect_count_add(self, add: int):
        self._detect_count += add

    def detect_count_rst(self):
        self._detect_count = 0

    @property
    def detector_state(self):
        return self._detector_state

    @detector_state.setter
    def detector_state(self, _state: bool):
        if self._detector_state is not _state:
            self._detector_state = _state

    def _put_text(self, _frame: np.ndarray, detection: np.ndarray) -> np.ndarray:
        left = detection[3] * self._frame_size[1]  # cols
        top = detection[4] * self._frame_size[0]  # rows
        label = self._label(detection)
        idx = int(detection[1])
        y = top - 15 if top - 15 > 15 else top + 15
        cv.putText(_frame, label, (int(left), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   self._classes.pick_color(index=idx), 1)

        return _frame

    def _detector(self, **kwargs) -> tuple:
        self.certainty = kwargs.get("certainty", 30)
        _network_output = kwargs.get("network_output", None)

        if _network_output is not None and self._wanted_score is not None:

            for detection in _network_output[0, 0, :, :]:
                _score = float(detection[2])
                if _score > self._wanted_score:
                    if self._label(detection) in self._what_to_detect:
                        # print(self._detect_count)
                        self.detect_count_add(1)
                        if self._detect_count == self._detect_threshold:
                            self.detect_count_rst()
                            self.detector_state = True

                        _newFrame = self._bounded_box(self._frame, detection)
                        _newFrame = self._put_text(_newFrame, detection)

                        return self.detector_state, _newFrame
                    else:
                        self.detect_count_add(1)
                        if self._detect_count == self._detect_threshold:
                            self.detect_count_rst()
                            self._detector_state = False

                        return self._detector_state, self._frame
                else:
                    self._detector_state = False

                    return self._detector_state, self._frame  # TODO add logging
        else:
            pass  # TODO add logging

    def detect(self, **kwargs) -> tuple:
        self._frame = kwargs.get("frame", None)
        _what_test = kwargs.get("what_to_detect", None)
        self._detect_threshold = kwargs.get("detect_threshold", 1)
        if isinstance(_what_test, list):
            self._what_to_detect = _what_test
        else:
            pass  # TODO add logging
        self.set_input()
        return self._detector(network_output=self.forward(), certainty=40.0)
