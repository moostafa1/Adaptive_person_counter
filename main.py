import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import *
import subprocess
from capture_img import take_capture

# import cvzone


model=YOLO('yolov8s.pt')



# def is_point_on_line(x1, y1, x2, y2, x, y):
#     if (y - y1) == ((y2 - y1) / (x2 - x1)) * (x - x1):
#         return True
#     return False



# def is_point_on_line(x1, y1, x2, y2, x, y):
#     if x2 - x1 == 0:  # vertical line
#         return x == x1
#     else:
#         return abs((y - y1) - ((y2 - y1) / (x2 - x1)) * (x - x1)) < 1e-6  # 1e-6 is used to handle floating point inaccuracies


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    min_x1, max_x1 = min(line1[0][0], line1[1][0]), max(line1[0][0], line1[1][0])
    min_y1, max_y1 = min(line1[0][1], line1[1][1]), max(line1[0][1], line1[1][1])
    min_x2, max_x2 = min(line2[0][0], line2[1][0]), max(line2[0][0], line2[1][0])
    min_y2, max_y2 = min(line2[0][1], line2[1][1]), max(line2[0][1], line2[1][1])

    if min_x1 <= x <= max_x1 and min_y1 <= y <= max_y1 and min_x2 <= x <= max_x2 and min_y2 <= y <= max_y2:
        return True
    return False


# to draw the line
flag = 0
line_coordinates = []
def draw_line(event, x, y, flags, param):
    global flag, line_coordinates
    if event == cv2.EVENT_MOUSEMOVE:
        print((x, y), f'coordinates = {line_coordinates}')

    if event == cv2.EVENT_LBUTTONDBLCLK and flag == 0:
        flag = 1
        line_coordinates.append((x, y))
    elif event == cv2.EVENT_LBUTTONDBLCLK and flag == 1:
        line_coordinates.append((x, y))
        flag = 2
    if event == cv2.EVENT_RBUTTONDOWN and len(line_coordinates):
        line_coordinates = []
        flag = 0



# points = []
# to get specific reigon of area to count people inside
# flag2 = 0
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
#     # Draw the polygon if there are at least 2 points
#     if len(points) > 1:
#         cv2.polylines(frame, [np.array(points)], False, (0, 255, 0), 2)
        # cv2.imshow("Image", img)
    # if len(points) > 2:
    #     flag2 = 1



# # for drone
# # take capture to count people inside a specific region of it
def take_capture(frame, points):
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # img = frame.copy()
        cv2.imwrite(r'counted images\to_count.jpg', frame)
        frame = cv2.imread(r'counted images\to_count.jpg')
        mask = np.zeros_like(frame)
        cv2.namedWindow('draw_polygon')
        cv2.setMouseCallback('draw_polygon', draw_polyline)
        # while True:
        # Display the image in the window
        cv2.imshow('draw_polygon', frame)
        if cv2.waitKey(1) & 0xFF == ord('d') and len(points) > 2:
            cv2.destroyWindow(frame)
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



# for gate
# determine if the people are got in or out a gate according to boundary_line drawn
def boundary_line(frame, line_coordinates):
    if len(line_coordinates) == 2:
        cv2.line(frame, line_coordinates[0], line_coordinates[1], (0,255,255), 2)
        # cv2.polylines(frame, [np.array(line_coordinates)], isClosed=False, color=(0, 255, 0), thickness=2)
    cv2.namedWindow('Adaptive human counter')
    cv2.setMouseCallback('Adaptive human counter', draw_line)




# write information over frames
def show_count(frame, num_persons, entered, get_out):
    cv2.putText(frame, 'Count: ', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(frame, str(num_persons), (160, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(frame, 'Entered: ', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(frame, str(entered), (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(frame, 'Leaved: ', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(frame, str(get_out), (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)






option = int(input("Enter (0 for drone, 1 for gate): "))

cap=cv2.VideoCapture('people.mp4')   # 'vtest.avi'

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)

count=0
persondown={}
tracker=Tracker()
counter1=[]

personup={}
counter2=[]
# cy1=194
# cy2=220
# offset=6
entered = 0
get_out = 0
cropping = []
# Add a dictionary to store the previous centroid positions
prev_centroids = {}

frame_order = 0

while True:
    ret,frame = cap.read()
    if not ret:
        break
#    frame = stream.read()

    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))


    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]

    num_persons = 0
    for index,row in px.iterrows():
#        print(row)

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])

        c=class_list[d]
        if 'person' in c:
            num_persons += 1
            list.append([x1,y1,x2,y2])


    bbox_id=tracker.update(list)

    for bbox in bbox_id:
        # print('bbox', bbox)
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        # Get the previous centroid position for this id
        prev_cx, prev_cy = prev_centroids.get(id, (cx, cy))

        # Determine the direction of movement
        direction = cy - prev_cy

        # Store the current centroid position for the next frame
        prev_centroids[id] = (cx, cy)

        mid_x = x3 + (x4 - x3) // 2
        cv2.circle(frame, (cx,cy), 4, (255,0,255), -1)
        # Draw a vertical line at the midpoint from the top to the bottom of the rectangle
        cv2.line(frame, (mid_x, y3), (mid_x, y4), (0, 0, 255), 2)

        if len(line_coordinates) >= 2:
            if line_intersection(line_coordinates, [(mid_x, y3), (mid_x, y4)]):
                if direction > 0:  # Moving downwards
                    entered += 1
                else:  # Moving upwards
                    get_out += 1



    if option == 0:
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(r'counted images\to_count.jpg', frame)
            # subprocess.Popen(["python", "capture_img.py"])
            take_capture(frame, points)
        # if frame != 0:


    elif option == 1:
        boundary_line(frame, line_coordinates)
        show_count(frame, num_persons, entered, get_out)
        cv2.imshow("Adaptive human counter", frame)



        # cv2.polylines(frame, [np.array(line_coordinates)], isClosed=False, color=(0, 255, 0), thickness=2)
        # cv2.polylines(frame, [np.array(line_coordinates)], isClosed=False, color=(0, 255, 0), thickness=2)

    # Break the loop if the 'q' key is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break




    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
