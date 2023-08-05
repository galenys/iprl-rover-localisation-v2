import requests
from requests.auth import HTTPDigestAuth
import cv2
import cv2.aruco as aruco
import numpy as np
from time import sleep

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

url = "http://192.168.1.1/osc/commands/execute"
username = "THETAYP00153381.OSC"
password = "00153381"

payload = {
    "name": "camera.getLivePreview"
}

headers = {
    "Content-Type": "application/json;charset=utf-8"
}

response = requests.post(url, auth=HTTPDigestAuth(username, password), json=payload, headers=headers, stream=True)

# We would load this up with the information given on the day of the contest
marker_preset_positions_and_directions = {
    8: np.array([[0, 0, 0], [1, 0, 0]])
}

if response.status_code == 200:
    bytes_ = bytes()
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            bytes_ += chunk
            a = bytes_.find(b'\xff\xd8')
            b = bytes_.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_[a:b+2]
                bytes_ = bytes_[b+2:]
                img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                markerCorners, markerIds, rejectedCandidates = aruco.detectMarkers(img, dictionary)
                # print(markerCorners, markerIds)
                img = aruco.drawDetectedMarkers(img, markerCorners, markerIds)
                cv2.imshow("Preview", img)
                if cv2.waitKey(1) == 27:
                    break
else:
    print("Error: ", response.status_code)

cv2.destroyAllWindows()
