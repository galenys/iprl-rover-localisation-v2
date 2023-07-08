import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

while(True):
    ret, frame = cap.read()
    markerCorners, markerIds, rejectedCandidates = aruco.detectMarkers(frame, dictionary)
    frame = aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
