from flask import Flask, render_template, Response
from subprocess import Popen, PIPE

import cv2

app = Flask(__name__ )

shape_predictor_path = '/Users/pranavdhawan/Projects/blinkit/shape_predictor_68_face_landmarks.dat'
gaze_model_path = '/Users/pranavdhawan/Projects/blinkit/models/gazev3.1.h5'
blink_model_path = '/Users/pranavdhawan/Projects/blinkit/models/blinkdetection.h5'


@app.route('/')
def index():
    blink_process = Popen(['bash', 'run_blink_detection.sh'], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5555', debug=True)

