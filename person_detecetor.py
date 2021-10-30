import source
import detector
import rest_api
import cv2 as cv
from flask import Flask, render_template, Response

app = Flask(__name__)
# src = source.Source(source=0)
src = source.Source(source="rtsp://admin:XPCWU3ZbiERSNav1v3Xz@192.168.2.9:554/live/video/profile1")
detect = detector.Detector(trainedModel="frozen_inference_graph.pb",
                           modelConfig="ssd_mobilenet_v2_coco_2018_03_29.pbtxt")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(person_detector(), mimetype='multipart/x-mixed-replace; boundary=frame')


body = {
    "state": "clear",
    "old_state": "detected",
    "attributes": {
        "detected": "person",
        "location": "outside"
    }
}


def notify_ha(state: str):
    person_api = rest_api.Api(base_url="http://homeassistant:8123/api/", api="states/sensor.person_detector")
    person_api.header = ("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJmYzJlZWI1N2ZmMDE0MDc5YmE4OGE1NDQ0NWU"
                         "4NzlkYiIsImlhdCI6MTYzMzg0NjAzMCwiZXhwIjoxOTQ5MjA2MDMwfQ.hgpZZM8l90WZjAXMAWm6RImPQVyN"
                         "YnkwRV4pBvklTv4")

    body['state'] = state
    body['location'] = "outside"
    if state != body['old_state']:
        person_api.post(data=body)
    body['old_state'] = state


def person_detector():
    # src = source.Source(source=0)
    # src = source.Source(source="rtsp://admin:XPCWU3ZbiERSNav1v3Xz@192.168.2.9:554/live/video/profile1")
    # detect = detector.Detector(trainedModel="frozen_inference_graph.pb",
    #                            modelConfig="ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

    previous_state: bool = False
    for frame in src.get_frame:
        found, img = detect.detect(frame=frame, what_to_detect=["person"], detect_threshold=5)
        print(found)
        if found:
            print("found person")
            if found is not previous_state:
                previous_state = found
                print("notify HA of detection")
                notify_ha(state="detected")
        else:
            if found is not previous_state:
                previous_state = found
                print("notify HA of clear")
                notify_ha(state="clear")
        ret, buffer = cv.imencode('.jpg', img)
        image = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result

        # cv.imshow("video", frame)
        # if cv.waitKey(50) & 0xFF == ord('q'):
        #     break


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
    # person_detector()
