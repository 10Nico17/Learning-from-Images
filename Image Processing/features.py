import cv2

cap = cv2.VideoCapture(0)
cv2.namedWindow('Learning from images: SIFT feature visualization')
sift = cv2.SIFT_create()
while True:
    ret, frame = cap.read()
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints = sift.detect(gframe, None)
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('Learning from images: SIFT feature visualization', frame_with_keypoints)
    if cv2.waitKey(1) != -1:
        break

cap.release()
cv2.destroyAllWindows()