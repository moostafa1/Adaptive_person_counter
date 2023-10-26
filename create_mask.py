import cv2
import numpy as np

# to get specific reigon of area to count people inside
flag2 = 0
points = []
def draw_polyline(event, x, y, flags, param):
    global flag2, points
    if event == cv2.EVENT_MOUSEMOVE:
        print((x, y), f'points = {points}')

    if event == cv2.EVENT_LBUTTONDBLCLK:
        points.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN and len(points):
        points = []
    if len(points) > 2:
        flag2 = 1


img = cv2.imread(r'counted images\to_count.jpg')
mask = np.zeros_like(img)
cv2.namedWindow('draw_polygon')
cv2.setMouseCallback('draw_polygon', draw_polyline)


# while True:
    # Display the image in the window
cv2.imshow('draw_polygon', img)
if cv2.waitKey(0) & 0xFF == ord('d') and flag2:
    # cv2.polylines(mask, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.fillPoly(mask, [np.array(points)], color=(255, 255, 255))
    points = []
    flag2 = 0

cv2.imshow('mask', mask)

# Bitwise-AND mask and original image
result = cv2.bitwise_and(img, mask, mask=None)
# result = cv2.bitwise_or(img, mask, mask=None)
cv2.imshow('result', result)


cv2.waitKey(0)
cv2.destroyAllWindows()
