import requests
from requests.auth import HTTPDigestAuth
import cv2
import cv2.aruco as aruco
import numpy as np

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

url = "http://192.168.1.1/osc/commands/execute"
username = "THETAYP00153381.OSC"
password = "00153381"

payload = {
    "name": "camera.getLivePreview"
}

headers = {
    "Content-Type": "application/json;charset=utf-8"
}

response = requests.post(url, auth=HTTPDigestAuth(username, password), json=payload, headers=headers, stream=True, verify=False)

# TODO: Calibrate camera
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
distortion_coefficients = np.zeros((4, 1))

# We would load this up with the information given on the day of the contest
marker_preset_positions_and_directions = {
    8: np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
    23: np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
}
parameters = cv2.aruco.DetectorParameters()

def camera_pose_from_marker(rvecs, tvecs, known_marker_position, known_marker_orientation):
    # Flatten the arrays to shape (3,)
    rvecs = rvecs.flatten()
    tvecs = tvecs.flatten()
    known_marker_position = known_marker_position.flatten()
    known_marker_orientation = known_marker_orientation.flatten()

    # Convert the rotation vector to a rotation matrix
    R_marker2cam, _ = cv2.Rodrigues(rvecs)
    
    # Apply known marker orientation (if provided)
    R_known_marker_orientation, _ = cv2.Rodrigues(known_marker_orientation)
    R_marker2world = np.dot(R_known_marker_orientation, np.transpose(R_marker2cam))
    
    # Invert the transformation
    R_cam2marker = np.transpose(R_marker2cam)
    t_cam2marker = -np.dot(R_cam2marker, tvecs.reshape(-1, 1))  # tvecs must be column vector for dot product
    
    # Calculate camera pose in world coordinate system
    R_cam2world = np.transpose(R_marker2world)
    t_cam2world = np.dot(-R_cam2world, known_marker_position.reshape(-1, 1)) + t_cam2marker  # known_marker_position must be column vector for dot product
    
    return R_cam2world, t_cam2world.flatten()  # flatten t_cam2world for consistency

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
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, distortion_coefficients)
    
    return rvecs, tvecs

def most_likely_pose(poses):
    # Assume len(poses) > 1

    # Can be fine tuned
    epsilon = 1
    highest_similar_position_count = -1
    most_likely = None
    for i, (position1, orientation1) in enumerate(poses):
        similar_positions = 0
        for j, (position2, _) in enumerate(poses):
            if i != j and np.abs(position1 - position2) < epsilon:
               similar_positions += 1 
        if similar_positions > highest_similar_position_count:
            highest_similar_position_count = similar_positions
            most_likely = (position1, orientation1)
    return most_likely

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

                markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)
                img = aruco.drawDetectedMarkers(img, markerCorners, markerIds)
                
                poses = []
                if markerIds is not None:
                    # print(markerIds)
                    for i, marker_id in enumerate(markerIds.ravel()):
                        known_marker_pose = marker_preset_positions_and_directions.get(marker_id, None)
                        if known_marker_pose is not None:

                            known_marker_position, known_marker_orientation = known_marker_pose[0], known_marker_pose[1]
                            rvecs, tvecs = detect_and_estimate_pose(img, markerCorners, markerIds)

                            R_cam, t_cam = camera_pose_from_marker(rvecs, tvecs, known_marker_position, known_marker_orientation)
                            euler_angles = rotation_matrix_to_euler_angles(R_cam)
                            poses.append((t_cam, euler_angles))
                            # print(t_cam, euler_angles)
                if poses:
                    # This is a tuple of (position, orientation)
                    # This is where we would send it over the network. Otherwise send None
                    print(most_likely_pose(poses))

                cv2.imshow("Preview", img)
                if cv2.waitKey(1) == 27:
                    break
else:
    print("Error: ", response.status_code)

cv2.destroyAllWindows()
