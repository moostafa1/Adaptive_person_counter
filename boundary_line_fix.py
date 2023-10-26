import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import *
import math
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




def check_intersection(circle_center, radius, line_point1, line_point2):
    x0, y0 = circle_center
    x1, y1 = line_point1
    x2, y2 = line_point2

    numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    try:
        distance = numerator / denominator
        return distance <= radius
    except:
        return 0





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







# to count the persons in the frame
def count_persons(frame, class_list, is_video=True):
    tracker=Tracker()
    results=model.predict(frame)
    a = results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    list=[]

    num_persons = 0
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])

        # c = class_list[d]
        if 'person' in class_list[d]:
            num_persons += 1
            list.append([x1,y1,x2,y2])

    if is_video:
        bbox_id = tracker.update(list)
        return bbox_id, num_persons
    else:
        return num_persons




def get_direction(line_coordinates, mid_x, y3, y4, person_down_left, counter1, person_up_right, counter2,
                    circle_center, direction_x, direction_y, radius):
    direction = ''
    counted_ids = set()
    if len(line_coordinates) >= 2:
        # if line_intersection(line_coordinates, [(mid_x, y3), (mid_x, y4)]):
        if check_intersection(circle_center, radius, line_coordinates[0], line_coordinates[1]):
            if id not in counted_ids:  # Only count if the id has not been counted in this frame
                # if direction > 0:  # Moving downwards
                #     entered += 1
                # else:  # Moving upwards
                #     get_out += 1


                # Determine the overall direction
                if abs(direction_x) > abs(direction_y):
                    if direction_x > 0:
                        person_up_right[id] = circle_center
                        counter2.append(id)
                        # direction = 'right'
                    else:
                        person_down_left[id] = circle_center
                        counter1.append(id)
                        # direction = 'left'
                elif abs(direction_y) > abs(direction_x):
                    if direction_y > 0:
                        person_down_left[id] = circle_center
                        counter1.append(id)
                        # direction = 'down'
                    else:
                        person_up_right[id] = circle_center
                        counter2.append(id)
                # if id in person_down_left:
                #     if counter1.count(id)==0:
                #         counter1.append(id)
                # elif id in person_up_right:
                #     if counter2.count(id)==0:
                #         counter2.append(id)
                #         direction = 'up'
                else:  # diagonal movement
                    if direction_x > 0 and direction_y > 0:
                        # direction = 'down-right'
                        person_up_right[id] = circle_center
                        counter2.append(id)
                    elif direction_x > 0 and direction_y < 0:
                        # direction = 'up-right'
                        person_up_right[id] = circle_center
                        counter2.append(id)
                    elif direction_x < 0 and direction_y > 0:
                        # direction = 'down-left'
                        person_down_left[id] = circle_center
                        counter1.append(id)
                    else:
                        # direction = 'up-left'
                        person_down_left[id] = circle_center
                        counter1.append(id)
                # counted_ids.add(id)  # Add the id to the counted ids
    return direction, counted_ids









class ImageDrawer:
    def __init__(self):
        self.points = []
        self.img = None
    # to get specific reigon of area to count people inside
    # points = []
    def draw_polyline(self, event, x, y, flags, param):
        global flag2, points
        if event == cv2.EVENT_MOUSEMOVE:
            print((x, y), f'points = {self.points}')

        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.points.append((x, y))
            cv2.circle(self.img, (x, y), 5, (0, 0, 255), -1)
        if event == cv2.EVENT_RBUTTONDOWN and len(self.points):
            self.points = []
        # Draw the polygon if there are at least 2 points
        if len(self.points) > 1:
            cv2.polylines(self.img, [np.array(self.points)], False, (0, 255, 0), 2)



    def main(self, video_name):
        cap=cv2.VideoCapture(video_name)   #  'vtest.avi' 'people.mp4'

        my_file = open("coco.txt", "r")
        data = my_file.read()
        class_list = data.split("\n")
        #print(class_list)

        count=0
        saved_count = 0

        person_down_left = {}
        counter1 = [] # set() #

        person_up_right = {}
        counter2 = [] # set() #

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

            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # call the model to predict the number of people in the image
            bbox_id, num_persons = count_persons(frame, class_list, is_video=True)
            # # Add a set to store the counted ids for each frame
            # counted_ids = set()
            # Define the buffer zone
            # buffer_zone = 10  # pixels
            # line_coordinates_buffer = [(x-buffer_zone, y-buffer_zone) for x, y in line_coordinates]

            for bbox in bbox_id:
                # print('bbox', bbox)
                x3,y3,x4,y4,id=bbox
                cx=int(x3+x4)//2
                cy=int(y3+y4)//2

                # Get the previous centroid position for this id
                prev_cx, prev_cy = prev_centroids.get(id, (cx, cy))

                # Determine the direction of movement
                direction_x = cx - prev_cx
                direction_y = cy - prev_cy

                # Store the current centroid position for the next frame
                prev_centroids[id] = (cx, cy)
                mid_x = x3 + (x4 - x3) // 2
                direction, counted_ids = get_direction(line_coordinates, mid_x, y3, y4, person_down_left,
                                                        counter1, person_up_right, counter2, (cx, cy), direction_x, direction_y, 2)

                cv2.circle(frame, (cx,cy), 5, (255,0,255), -1)
                cv2.putText(frame, direction, (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

            # At the end of each frame, clear the counted ids
            # counted_ids.clear()

                # Draw a vertical line at the midpoint from the top to the bottom of the rectangle
                # cv2.line(frame, (mid_x, y3), (mid_x, y4), (0, 0, 255), 2)


            # cv2.line(frame,(3,194),(1018,194),(0,255,0),2)
            # cv2.line(frame,(5,220),(1019,220),(0,255,255),2)

            if len(line_coordinates) == 2:
                cv2.line(frame, line_coordinates[0], line_coordinates[1], (0,255,255), 2)
                # cv2.polylines(frame, [np.array(line_coordinates)], isClosed=False, color=(0, 255, 0), thickness=2)

            # counting people in defined area of region
            if cv2.waitKey(1) & 0xFF == ord('c'):
                self.img = frame.copy()
                saved_count += 1
                cv2.imwrite(f'counted images\\{saved_count}.jpg', frame)
                self.img = cv2.imread(f'counted images\\{saved_count}.jpg')
                mask = np.zeros_like(self.img)
                cv2.namedWindow('draw_polygon')
                cv2.setMouseCallback('draw_polygon', self.draw_polyline)

                while True:
                    # Display the image in the window
                    cv2.imshow('draw_polygon', self.img)
                    if cv2.waitKey(1) & 0xFF == ord('d') and len(self.points) > 2:
                        # scv2.destroyWindow(img)
                        self.points.append(self.points[0])
                        cv2.polylines(self.img, [np.array(self.points)], isClosed=True, color=(0, 255, 0), thickness=2)
                        cv2.fillPoly(mask, [np.array(self.points)], color=(255, 255, 255))
                        self.points = []
                        # Bitwise-AND mask and original image
                        result = cv2.bitwise_and(self.img, mask, mask=None)
                    elif cv2.waitKey(1) & 0xFF == ord('s'):
                        # cv2.destroyWindow(img)
                        cnt_persons = count_persons(result, class_list, is_video=False)
                        cv2.putText(result, 'Count: ', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
                        cv2.putText(result, str(cnt_persons), (160, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
                        cv2.imshow('result', result)
                        cv2.imwrite(f'masked images\\{saved_count}.jpg', result)
                    elif cv2.waitKey(1) & 0xFF == 27:
                        cv2.destroyAllWindows()
                        break
                    elif len(self.points) and cv2.waitKey(1) == 8:
                        self.points.pop()
                        img = cv2.imread(f'counted images\\{saved_count}.jpg')
                        cv2.polylines(img, [np.array(self.points)], False, (0, 255, 0), 2)
                        for pt in self.points:
                            cv2.circle(img, pt, 5, (0, 0, 255), -1)
                        self.img = img


            cv2.namedWindow('Adaptive human counter')
            cv2.setMouseCallback('Adaptive human counter', draw_line)

            cv2.putText(frame, 'Count: ', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
            cv2.putText(frame, str(num_persons), (160, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
            cv2.putText(frame, 'Entered: ', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
            cv2.putText(frame, str(len(counter1)), (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
            cv2.putText(frame, 'Leaved: ', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
            cv2.putText(frame, str(len(counter2)), (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
            cv2.imshow("Adaptive human counter", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()



def cv_project(video_name):
    drawer = ImageDrawer()
    drawer.main(video_name)


if __name__ == "__main__":
    cv_project('people.mp4')
