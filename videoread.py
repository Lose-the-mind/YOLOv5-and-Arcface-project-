import cv2
ip = '172.16.25.245'
cap = cv2.VideoCapture("rtsp://172.16.25.245/13")
ret, frame = cap.read()
cv2.namedWindow(ip, 0)
cv2.resizeWindow(ip, 500, 300)
while ret:
    ret, frame = cap.read()
    cv2.imshow(ip, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cv2.destroyAllWindows()
cap.release()
