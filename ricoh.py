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

def estimate_distance(marker_corner):
    corner_spread = np.max(np.linalg.norm(marker_corner[0] - marker_corner[2]), np.linalg.norm(marker_corner[1] - marker_corner[3]))
    distance = 1.0 / corner_spread
    return distance

def estimate_position_and_orientation(marker_corners, marker_ids):
    potential_positions = []
    for i in range(len(marker_ids)):
        marker_id = marker_ids[i]
        marker_corner = marker_corners[i]
        if marker_id in marker_preset_positions_and_directions:
            position, direction = marker_preset_positions_and_directions[marker_id]
            distance = estimate_distance(marker_corner)
            estimated_position = position + distance * direction
            weight = 1.0 / np.abs(estimated_position)
            potential_positions.append((weight, estimated_position, -direction))
            
    # Weighted average of positions and directions
    sum_positions = np.zeros(3)
    sum_directions = np.zeros(3)
    for weight, position, direction in potential_positions:
        sum_positions += weight * position
        sum_directions += weight * direction

    return sum_positions / np.sum(weight), sum_directions / np.sum(weight)

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
