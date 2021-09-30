import base64
import os
import time

import cv2
import numpy
from flask import Flask, request, url_for, render_template, flash
from werkzeug.utils import redirect

import app as app
from NumberPlateRecognition.code.MainPrediction import predict

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


@app.route('/')
def home():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        filename = file.filename
        if file and allowed_file(file.filename):
            npimg = numpy.fromstring(file.read(), numpy.uint8)  # convert string data to numpy array
            img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)  # convert numpy array to image
            path = os.path.join(app.config['UPLOAD_FOLDER']) + filename
            start = time.time()
            cv2.imwrite(path, img)
            try:
                string, plate = predict(img)
                print("ki tu: ", string)
            except:
                flash('Error: Có lỗi xảy ra! Kiểm tra ảnh đầu vào')
                return redirect(request.url)
            if string is None and plate is None:
                flash('Error: Có lỗi xảy ra! Vui lòng thử lại')
                return redirect(request.url)
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
                if string == '':
                    return_data['bsx'] = 'Không thể nhận diện !!!'
                return render_template('index.html', predict=return_data['bsx'], filename=filename)
        else:
            flash('Allowed image types are ->png, jpg, jpeg')
            return redirect(request.url)
    return render_template('index.html')


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    app.run()
