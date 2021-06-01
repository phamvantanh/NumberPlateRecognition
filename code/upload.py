import base64
import os
import time

import app as app
import cv2
import numpy
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename, redirect

from NumberPlateRecognition.code.Main1 import predict

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        # file1 = request.files['file1']
        # print(type(file1))
        # img = cv2.imread(file1)
        # cv2.imshow("img", img)
        file2 = request.files['file1'].read()
        file = request.files['file1']
        filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # convert string data to numpy array
        npimg = numpy.fromstring(file2, numpy.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        path = os.path.join(app.config['UPLOAD_FOLDER']) + filename
        # print(path)
        start = time.time()
        cv2.imwrite(path, img)
        try:
            string, plate = predict(img)
            print("ki tu: ", string)
        except:
            return "a", 406
        if string is None and plate is None:
            # return_data = {
            #     'bsx': None,
            #     'image': None,
            #     'time': None
            # }
            return "b", 406
        else:
            _, im_arr = cv2.imencode(".jpg", plate)
            im_bytes = im_arr.tobytes()
            img = base64.b64encode(im_bytes)
            end = time.time()
            return_data = {
                'bsx': "Result: " + string,
                'image': img.decode('utf-8'),
                'time': end - start
            }
            return render_template('index.html', predict=return_data['bsx'], filename=filename)
        return 'ok'
    return render_template('index.html')


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    app.run()
