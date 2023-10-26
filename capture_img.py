import cv2
import numpy as np




points = []
def draw_polyline(event, x, y, flags, param):
    global flag2, points
    if event == cv2.EVENT_MOUSEMOVE:
        print((x, y), f'points = {points}')

    if event == cv2.EVENT_LBUTTONDBLCLK:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    if event == cv2.EVENT_RBUTTONDOWN and len(points):
        points = []
    # Draw the polygon if there are at least 2 points
    # if len(points) > 1:
    #     cv2.polylines(frame, [np.array(points)], False, (0, 255, 0), 2)




# for drone
# take capture to count people inside a specific region of it
def take_capture(frame, points):
    # if cv2.waitKey(1) & 0xFF == ord('c'):
    #     cv2.imwrite(r'counted images\to_count.jpg', frame)
        # img = frame.copy()
    frame = cv2.imread(r'counted images\to_count.jpg')
    mask = np.zeros_like(frame)
    cv2.namedWindow('draw_polygon')
    cv2.setMouseCallback('draw_polygon', draw_polyline, frame)
    # while True:
    # Display the image in the window
    cv2.imshow('draw_polygon', frame)
    if cv2.waitKey(1) & 0xFF == ord('d') and len(points) > 2:
        #cv2.destroyWindow(frame)
        cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.fillPoly(mask, [np.array(points)], color=(255, 255, 255))
        points = []
        flag2 = 0
        # Bitwise-AND mask and original image
        result = cv2.bitwise_and(frame, mask, mask=None)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.destroyWindow('draw_polygon')
        cv2.imshow('result', result)
        cv2.imwrite(r'counted images\masked.jpg', result)
        # cv2.waitKey(1000)
    if cv2.waitKey(1) & 0xFF == 27:
        return 0
    # return frame
