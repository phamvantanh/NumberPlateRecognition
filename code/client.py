import requests
import base64
import json
import cv2
import numpy as np
BASE = "http://localhost:8000/bsx"

path_img = "./test-image.jpg"
# path_img = "./Example/image-test5.jpg"
with open(path_img, "rb") as f:
    im_b64 = base64.b64encode(f.read())
# print(len(im_b64))
response = requests.post(BASE, data={
    'image' : im_b64,
})
data =json.loads(response.text)
img = data['image']
img = base64.b64decode(img)
img = np.frombuffer(img, dtype=np.uint8)
img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
cv2.imshow("", img)
cv2.waitKey()
cv2.destroyAllWindows()

# cv2.destroyAllWindows()
