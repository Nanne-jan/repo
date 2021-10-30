import cv2 as cv


class Source:
    def __init__(self, **kwargs):
        self.stream = kwargs.get("source", None)
        self._capobj = None

    @property
    def stream(self) -> str:
        return self._stream

    @stream.setter
    def stream(self, source: str):
        self._stream = source
    
    def _open(self):
        _capobj = cv.VideoCapture(self.stream)
        if _capobj.isOpened():
            return _capobj
        else:
            print("Cannot open source stream")
            exit(1)

    @property
    def get_frame(self):
        if self._capobj is None:
            self._capobj = self._open()
        while self._capobj.isOpened():
            success, _frame = self._capobj.read()
            if success:
                yield _frame
            else:
                return

    @property
    def get_size(self) -> tuple:
        return self._size()

    def _size(self) -> tuple:
        self._open()
        self.height = int(self._capobj.get(3))
        self.width = int(self._capobj.get(4))

        return self.width, self.height


if __name__ == "__main__":
    # vid = Source(source="rtsp://admin:XPCWU3ZbiERSNav1v3Xz@192.168.2.9:554/live/video/profile1")
    vid = Source(source=1)
    for frame in vid.get_frame:
        cv.imshow("video", frame)
        if cv.waitKey(50) & 0xFF == ord('q'):
            break
