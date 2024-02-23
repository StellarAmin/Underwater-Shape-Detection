import cv2
import numpy as np
import time
from predict_shape import predict
current_point = (0, 0)
points = []
# cap = cv2.VideoCapture('tst20-9/red_star.mp4')
# cap = cv2.VideoCapture('tst20-9/blue_triangle.mp4')
cap = cv2.VideoCapture('red_star_2.mp4')
pause = False
def onClick(event, x, y, *args):
    global ref_point, current_point, mask_ready
    current_point = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))



def k_means(image, K):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    orig_shape = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    _, labels, (centers) = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(orig_shape)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    mask = cv2.medianBlur(mask, 7)
    con, hie = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    con_area = [cv2.contourArea(i) for i in con]
    con = con[con_area.index(max(con_area))]
    x,y, w, h = cv2.boundingRect(con)
    mask = mask[y:y+h, x:x+w]
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = cv2.resize(mask, (224, 224))
    return mask

cv2.namedWindow('Live')
cv2.setMouseCallback('Live', onClick)

orig_frame = None
while True:
    if not(pause):
        ret, frame = cap.read()
        orig_frame = frame.copy()
    frame = orig_frame.copy()

    if len(points) == 1:
        cv2.rectangle(frame, points[0], current_point, (0, 0, 255), 2)

    if len(points) == 2:
        print(points)
        img = frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]
        mask = k_means(img, 2)
        t1 = time.time()
        shape_cls = predict(mask)
        t2 = time.time()
        print("Predicted in ", (t2-t1))
        print(shape_cls)
        cv2.imshow('sa', mask)
    key = cv2.waitKey(1)
    if key == ord('p'):
        pause = not(pause)

    if key == ord('r'):
        current_point = (0, 0)
        points = []
        print('Points Reset')

    cv2.imshow('Live', frame)
