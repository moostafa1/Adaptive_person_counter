import cv2
import numpy as np


#array to save the points of shape
points = []  # polylines


# function to close the shape
def return_to_first_point():
    global points
    if len(points) > 2:
        points.append(points[0])
        cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)


# function to create mask to extract ROI of area in frame of video
def create_poly_mask():
    global points, frame
    mask = np.zeros_like(frame)
    if len(points) > 2:
        cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))
    masked_image = cv2.bitwise_and(frame, mask)
    return masked_image


# function to clear the shape
def clear_lines():
    global points
    points = []
    print("delete shape")





# Events of draw the shape


def click_event(event, x, y, flags, param):

    rect_start = None
    rect_end = None

#poly shape
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 0:
            return_to_first_point()

# rectangle shape



cv2.namedWindow('Image')
cv2.setMouseCallback('Image', click_event)

# read video
cap = cv2.VideoCapture('people.mp4')

while True:
    ret, frame = cap.read()

    #draw lines
    if len(points) > 1:
        cv2.polylines(frame, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)

    cv2.imshow('Image', frame)



    #some orders
    key = cv2.waitKey(1) & 0xFF

    #close running
    if key == ord('q'):
        break

    # create mask to extract area
    elif key == ord('s'):
        mask = create_poly_mask()
        cv2.imwrite("screen.png", mask)
        print("saved image")

    #clear the poly
    elif key == ord('c'):
        clear_lines()


cap.release()
cv2.destroyAllWindows()
