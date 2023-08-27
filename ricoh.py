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

# TODO: Calibrate camera
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
distortion_coefficients = np.zeros((4, 1))

# We would load this up with the information given on the day of the contest
marker_preset_positions_and_directions = {
    8: np.array([[0, 0, 0], [1, 0, 0]])
}
parameters = cv2.aruco.DetectorParameters()

def camera_pose_from_marker(rvecs, tvecs):
    # Convert the rotation vector to a rotation matrix
    R_marker2cam, _ = cv2.Rodrigues(rvecs[0])
    
    # Invert the transformation
    R_cam2marker = np.transpose(R_marker2cam)
    t_cam2marker = -np.dot(R_cam2marker, tvecs[0])
    
    return R_cam2marker, t_cam2marker

def rotation_matrix_to_euler_angles(R):
    # Compute yaw, pitch, roll
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])  # Pitch, Yaw, Roll

def detect_and_estimate_pose(image, corners, ids):
    # Draw detected markers (for visualization)
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    
    # Assuming the marker size is 0.15 meters (adjust as necessary)
    marker_size = 0.15
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, distortion_coefficients)
    
    return rvecs, tvecs

if response.status_code == 200:
    bytes_ = bytes()
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            bytes_ += chunk
            a = bytes_.find(b'\xff\xd8')
            b = bytes_.find(b'\xff\xd9')
            if a != -1 and b != -1:
                # Some byte stuff I don't understand, the bottom line is we have the image
                jpg = bytes_[a:b+2]
                bytes_ = bytes_[b+2:]
                img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                # Step 1: Detect markers
                markerCorners, markerIds, rejectedCandidates = aruco.detectMarkers(img, dictionary)
                img = aruco.drawDetectedMarkers(img, markerCorners, markerIds)
                if markerIds is None:
                    continue
                # Step 2: Get rotation and translation vectors
                rvecs, tvecs = detect_and_estimate_pose(img, markerCorners, markerIds)
                # print(rvecs, tvecs)
                # Step 3: Get camera pose
                R_cam, t_cam = camera_pose_from_marker(rvecs[0], tvecs[0])
                euler_angles = rotation_matrix_to_euler_angles(R_cam)
                print(t_cam, euler_angles)

                cv2.imshow("Preview", img)
                if cv2.waitKey(1) == 27:
                    break
else:
    print("Error: ", response.status_code)

cv2.destroyAllWindows()
