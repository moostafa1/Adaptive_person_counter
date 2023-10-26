import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone


model=YOLO('yolov8s.pt')


flag = 0
line_coordinates = [(0,0),(0,0)]
def draw_line(event, x, y, flags, param):
    print("draw_line")
    global flag, line, line_coordinates
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

    if event == cv2.EVENT_LBUTTONDBLCLK:
        flag = 1
        line_coordinates.append((x, y))
    elif event == cv2.EVENT_LBUTTONDBLCLK and flag == 1:
        line_coordinates.append((x, y))
        flag = 2
    if event == cv2.EVENT_RBUTTONDOWN and len(line_coordinates):
        line_coordinates = []
        flag = 0



# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :
#         point = [x, y]
#         print(point)



cv2.namedWindow('RGB')
#cv2.setMouseCallback('RGB', RGB)

cv2.setMouseCallback('RGB', draw_line)

#operto
cap=cv2.VideoCapture('vtest.avi') #F:/NTA/yolov8peoplecounter-main/vidp.mp4 ,vtest.avi


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)


count=0
tracker=Tracker()
counter1=[]

person_H_down={}
person_H_up={}
person_V_right={}
person_V_left={}
person_VI_right={}
person_VI_left={}

counter2=[]
#cy1=250

cy1=line_coordinates[0][1]

#cy2=300

cy2=line_coordinates[1][1]

once = 1

offset=200


lencounter1 = 0
lencounter2 = 0
# is_up = 0
# is_down = 0


while True:
    #print(545)
    ret,frame = cap.read()
    #print(54115)
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
    #print(px)
    list=[]

    sum=0
    for index,row in px.iterrows():
        #print(row)
        #print(index)


        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        #print(d)



        c=class_list[d]
        if 'person' in c:

            list.append([x1,y1,x2,y2])
            sum += 1

    bbox_id=tracker.update(list)
    # if len(line_coordinates) >= 2:
    #         cv2.line(frame, line_coordinates[0], line_coordinates[1], (0,255,255), 2)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        #cy=int(y3+y4)//2
        cy=int(y4)
        cv2.circle(frame,(cx,cy),8,(255,0,255),-1)
        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
        #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)

        # if line_coordinates[0][1]<(cy+offset) and line_coordinates[0][1]<(cy-offset):
        #     cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
        #     cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
        #     persondown[id]=(cx,cy)
        # if id in persondown:
        #     if line_coordinates[1][1]<(cy+offset) and line_coordinates[1][1]<(cy-offset):
        #         cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
        #         cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
        #         if counter1.count(id) == 0:
        #             counter1.append(id)
        #             lencounter1=counter1._len_()


        if len(line_coordinates) >= 2:
            cv2.line(frame, line_coordinates[0], line_coordinates[1], (0,255,255), 2)
            if(line_coordinates[0][0] >= line_coordinates[1][0]):
                linex1 =line_coordinates[1][0]
                linex2 =line_coordinates[0][0]
                liney1 =line_coordinates[1][1]
                liney2 =line_coordinates[0][1]
            else:
                linex1 =line_coordinates[0][0]
                linex2 =line_coordinates[1][0]
                liney1 =line_coordinates[0][1]
                liney2 =line_coordinates[1][1]
            # cv2.line(frame, (400,200), (700,500), (0,255,255), 2)
            # linex1 =400
            # linex2 =700
            # liney1 =200
            # liney2 =500

        # elif len(line_coordinates) > 2:
        #     line_coordinates = [line_coordinates[-1]]


            # if (line_coordinates[0][0] <= line_coordinates[1][0]):
            #     linex1 =line_coordinates[0][0]
            #     linex2 =line_coordinates[1][0]
            # else:
            #     linex1 =line_coordinates[1][0]
            #     linex2 =line_coordinates[0][0]

            # if (line_coordinates[0][1] <= line_coordinates[1][1]):
            #     liney1 =line_coordinates[0][1]
            #     liney2 =line_coordinates[1][1]
            # else:
            #     liney1 =line_coordinates[1][1]
            #     liney2 =line_coordinates[0][1]



            # #person in ( down or east )
            # if  ( cx in range(linex1, linex2 ) and  cy in range(0, liney1 ) )  or ( cx in range(0, linex1 ) and  cy in range(liney1, liney2) ):
            #     #cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
            #     #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
            #     persondown[id]=(cx,cy)

            # if id in persondown:
            #     if ( cx in range(linex1,linex2 ) and cy in range(liney1, liney2 +offset ) ) or  ( cx in range(linex1,linex2) and cy in range(liney1, liney2  ) ) :
            #         cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
            #         #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
            #         if counter1.count(id) == 0:
            #             counter1.append(id)
            #             lencounter1=counter1._len_()

            # #person out ( up and west )
            # if (cx in range(linex1, linex2 ) and cy in range(liney1, liney2 +500 )) or (cx in range(linex2, linex2 +offset) and cy in range(liney1, liney2 ) ):
            #     #cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,0),2)
            #     #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
            #     personup[id]=(cx,cy)


            # if id in personup:
            #     if (cx in range(linex1, linex2) and  cy in range(0, liney1 )) or (cx in range(0, linex1 ) and  cy in range(liney1, liney2)):
            #         cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
            #         #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
            #         if counter2.count(id) == 0:
            #             counter2.append(id)
            #             lencounter2=counter2._len_()





            # افقي
            if( liney1==liney2 ):
                # افقي نازل
                if cx in range(linex1, linex2 +1 ) and  cy in range(0, liney1 ) :
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
                    #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                    person_H_down[id]=(cx,cy)

                if id in person_H_down:
                    if cx in range(linex1,linex2 +1 ) and cy in range(liney2, liney2 +offset ):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                        #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                        if counter1.count(id) == 0:
                            counter1.append(id)
                            lencounter1=counter1._len_()


                #افقي طالع
                if cx in range(linex1, linex2 +1 ) and cy in range(liney2, liney2 +offset ):
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,0),2)
                    #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                    person_H_up[id]=(cx,cy)

                if id in person_H_up:
                    if cx in range(linex1, linex2 +1) and  cy in range(0, liney1 ):
                        #cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
                        #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                        if counter2.count(id) == 0:
                            counter2.append(id)
                            lencounter2=counter2._len_()




            # راسي
            elif( linex1 == linex2 ):
                # راسي يمين
                if cx in range(0, linex1 +1 ) and  cy in range(liney1, liney2 +1 ) :
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
                    #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                    person_V_right[id]=(cx,cy)

                if id in person_V_right:
                    if cx in range(linex1,linex2 +offset ) and cy in range(liney1, liney2 +1 ):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                        #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                        if counter1.count(id) == 0:
                            counter1.append(id)
                            lencounter1=counter1._len_()

                #راسي شمال
                if cx in range(linex1, linex2 +offset ) and cy in range(liney1, liney2 +1 ):
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,0),2)
                    #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                    person_V_left[id]=(cx,cy)

                if id in person_V_left:
                    if cx in range(0, linex1 +1) and  cy in range(liney1, liney2 +1 ):
                        #cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
                        #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                        if counter2.count(id) == 0:
                            counter2.append(id)
                            lencounter2=counter2._len_()


            # #  راسي مايل يمبن
            # elif( linex1 > linex2, liney1 < liney2 ):
            #     # راسي مايل يمبن
            #     if ( linex1 > cx > linex2 ) and  ( liney2 > cy > liney1 ):
            #         cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
            #         cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
            #         person_VI_right[id]=(cx,cy)

            #     if id in person_VI_right:
            #         if ( linex1 > cx > linex2 )  and (  liney2 > cy > liney1 ):
            #             cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,0),2)
            #             #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
            #             if counter1.count(id) == 0:
            #                 counter1.append(id)
            #                 lencounter1=counter1._len_()


            # #  راسي مايل شمال
            # elif( (cx in range(linex1,linex2 +1)) and (cy in range(liney1,liney2 +1)) and linex1 < linex2 and liney1 < liney2 ):
            #     # راسي مايل شمال
            #     linex= linex2 - linex1
            #     liney= liney2 - liney1
            #     linexd= linex2 / linex1
            #     lineyd= liney2 / liney1
            #     cx1= cx - linex
            #     cy1= cy - liney
            #     cx2= cx + linex
            #     cy2= cy + liney

            #     if (    linex1 > cx1 -1) and (liney1 > cy1 -1) :
            #         cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
            #         #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
            #         person_VI_right[id]=(cx,cy)

            #     if id in person_VI_right:
            #         if (linex2 < cx2 +1) and (liney2 < cy2 +1 ):
            #             cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,0),2)
            #             #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
            #             if counter1.count(id) == 0:
            #                 counter1.append(id)
            #                 lencounter1=counter1._len_()

            #     #راسي مايل شمال
            #     if (linex2 < cx2 +1) and (liney2 < cy2 +1):
            #         cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
            #         #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
            #         person_VI_left[id]=(cx,cy)

            #     if id in person_VI_left:
            #         if (linex1 > cx1 -1) and (liney1 > cy1 -1):
            #             cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
            #             #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
            #             if counter2.count(id) == 0:
            #                 counter2.append(id)
            #                 lencounter2=counter2._len_()



            #  راسي مايل شمال
            elif( (cx in range(linex1,linex2 +1)) and (cy in range(liney1,liney2 +1)) and linex1 < linex2 and liney1 < liney2 ):
                 # راسي مايل شمال
            #     linex= linex2 - linex1
            #     liney= liney2 - liney1
            #     linexd= linex2 / linex1
            #     lineyd= liney2 / liney1
            #     cx1= cx - linex
            #     cy1= cy - liney
            #     cx2= cx + linex
            #     cy2= cy + liney
                line_slope= (liney2 - liney1)/(linex2 - linex1)
                line_intercept = liney1 - line_slope * linex1

                if (cy > line_slope * cx + line_intercept) :
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
                    cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                    person_VI_right[id]=(cx,cy)

                if id in person_VI_right:
                    if (cy < line_slope * cx + line_intercept):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,0),2)
                        cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                        if counter1.count(id) == 0:
                            counter1.append(id)
                            lencounter1=counter1._len_()

                #راسي مايل شمال
                if (cy < line_slope * cx + line_intercept):
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
                    #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                    person_VI_left[id]=(cx,cy)

                if id in person_VI_left:
                    if (cy > line_slope * cx + line_intercept):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                        #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                        if counter2.count(id) == 0:
                            counter2.append(id)
                            lencounter2=counter2._len_()




            #  راسي مايل يمين
            elif( (cx in range(linex1,linex2 +1)) and (cy in range(liney2,liney1 +1)) and linex1 < linex2 and liney1 > liney2 ):
                 # راسي مايل يمين
            #     linex= linex2 - linex1
            #     liney= liney2 - liney1
            #     linexd= linex2 / linex1
            #     lineyd= liney2 / liney1
            #     cx1= cx - linex
            #     cy1= cy - liney
            #     cx2= cx + linex
            #     cy2= cy + liney
                line_slope= (liney1 - liney2)/(linex1 - linex2)
                line_intercept = liney1 - line_slope * linex1

                if (cy > line_slope * cx + line_intercept) :
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
                    cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                    person_VI_right[id]=(cx,cy)

                if id in person_VI_right:
                    if (cy < line_slope * cx + line_intercept):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,0),2)
                        cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                        if counter1.count(id) == 0:
                            counter1.append(id)
                            lencounter1=counter1._len_()


                if (cy < line_slope * cx + line_intercept):
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
                    #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                    person_VI_left[id]=(cx,cy)

                if id in person_VI_left:
                    if (cy > line_slope * cx + line_intercept):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                        #cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                        if counter2.count(id) == 0:
                            counter2.append(id)
                            lencounter2=counter2._len_()



            if once == 1:
                line_coordinates = []
                once = 2

    totalin = lencounter1 - lencounter2
    if totalin <= 0:
        totalin = 0
    else:
        totalin = totalin


    cv2.putText(frame,f'in = {lencounter1}',(20,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    cv2.putText(frame,f'out = {lencounter2}',(20,130),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    cv2.putText(frame,f'total in = {totalin}',(20,160),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)


    cv2.putText(frame,f'count = {sum}',(20,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    #cv2.line(frame,(300,cy1),(1018,cy1),(0,255,0),2)
    #cv2.line(frame,(300,cy2),(1019,cy2),(0,255,255),2)


    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
