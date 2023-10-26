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
cv2.namedWindow('draw_polygon')
cv2.setMouseCallback('draw_polygon', draw_polyline)


# while True:
    # Display the image in the window
cv2.imshow('draw_polygon', img)
if cv2.waitKey(0) & 0xFF == ord('d') and flag2:
    pts = cv2.polylines(img, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)
    print(pts)
    points = []
    flag2 = 0


cv2.destroyAllWindows()



# import cv2
# import numpy as np
#
# flag2 = 0
# points = []
#
# def draw_polyline(event, x, y, flags, param):
#     global flag2, points
#
#     if event == cv2.EVENT_MOUSEMOVE:
#         print((x, y), f'points = {points}')
#
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         points.append((x, y))
#
#     if event == cv2.EVENT_RBUTTONDOWN and len(points):
#         points = []
#
#     if len(points) > 2:
#         flag2 = 1
#
#
# img = cv2.imread(r'counted images\to_count.jpg')
# cv2.namedWindow('draw_polygon')
# cv2.setMouseCallback('draw_polygon', draw_polyline)
#
# while True:
#     cv2.imshow('draw_polygon', img)
#     if cv2.waitKey(1) & 0xFF == 13 and flag2:
#         cv2.polylines(img, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)
#         points = []
#         flag2 = 0
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
#         break
#
# cv2.destroyAllWindows()
