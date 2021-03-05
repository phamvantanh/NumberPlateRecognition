# import flaskApi
from PIL import Image
from io import BytesIO
import base64
from Main1 import predict
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import cv2
import time
import numpy as np
import requests

my_port = '8000'

# Khoi tao server
app = Flask(__name__)
CORS(app)


@app.route('/')
@cross_origin()
def index():
    return "Welcome to flask API"


# @app.route('/test', methods=["POST"])
# @cross_origin()

# def index2():
#     return "Welcome to flask API"


# khai bao xu ly request index
@app.route('/bsx', methods=['POST'])
@cross_origin()
def _hello_world():
    start = time.time()

    image = request.form.get('image')
    image = Image.open(BytesIO(base64.b64decode(image)))
    image.save('image1.jpg', 'JPEG')
    img_path = "image1.jpg"
    try:
        string, plate = predict(img_path)
        print("ki tu: ", string)
    except:
        return "", 406
    if string is None and plate is None:
        # return_data = {
        #     'bsx': None,
        #     'image': None,
        #     'time': None
        # }
        return "", 406
    else:
        _, im_arr = cv2.imencode(".jpg", plate)
        im_bytes = im_arr.tobytes()
        img = base64.b64encode(im_bytes)
        end = time.time()
        #if len(string) != 10 and len(string) != 12:
            #string = None
        return_data = {
            'bsx': string,
            'image': img.decode('utf-8'),
            'time': end - start
        }
        return return_data, 200


if __name__ == '__main__':
    app.run(debug=True, host='192.168.137.236', port=my_port)
    # app.run(debug=True, host='0.0.0.0', port=my_port)
