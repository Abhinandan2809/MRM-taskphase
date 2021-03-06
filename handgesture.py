import cv2
import numpy as np
import math
capture=cv2.VideoCapture(0)
while capture.isOpened():
 ret, frame=capture.read()
 cv2.rectangle(frame,(100,100) ,(400,400), (255,0,0),0)
 crop_image=frame[100:400, 100:400]
 blur=cv2.GaussianBlur(crop_image,(3,3),0)
 hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
 mask = cv2.inRange(hsv, np.array([108, 23,82]), np.array([179, 255, 255]))
 kernel=np.ones((7,7))
 opened=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
 gaussfiltered=cv2.GaussianBlur(opened,(5,5),0)
 ret,thresh=cv2.threshold(gaussfiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 cv2.imshow("Threshold", thresh)
 (contours, hierarchy) = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 try:
     contour=max(contours, key=lambda x: cv2.contourArea(x))
     x,y,w,h= cv2.boundingRect(contour)
     cv2.rectangle(crop_image, (x,y), (x+w,y+h),(0,0,255), 0)
     hull=cv2.convexHull(contour)
     drawhull = np.zeros(crop_image.shape, np.uint8)
     cv2.drawContours(drawhull, [contour], -1, (0, 255, 0), 0)
     cv2.drawContours(drawhull, [hull], -1, (0, 0, 255), 0)
     hull=cv2.convexHull(contour, returnPoints=False)
     defects=cv2.convexityDefects(contour, hull)
     count=0
     for i in range(defects.shape[0]):
         s, e, f, d = defects[i, 0]
         start = tuple(contour[s][0])
         end = tuple(contour[e][0])
         far = tuple(contour[f][0])

         a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
         b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
         c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
         angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
         if angle<=90:
             count+=1
             cv2.circle(crop_image, far, 1, [0, 0, 255], -1)
         cv2.line(crop_image,start,end,[0,255,0], 2)
     if count==0:
          cv2.putText(frame, "One", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
     elif count == 1:
          cv2.putText(frame, "Two", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
     elif count == 2:
          cv2.putText(frame, "Three", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
     elif count == 3:
          cv2.putText(frame, "Four", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
     elif count== 4:
          cv2.putText(frame, "Five", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
     else:
          pass
 except:
     pass
 cv2.imshow("Gesture", frame)
 all_image=np.hstack((drawhull, crop_image))
 cv2.imshow('Contours', all_image)
 if cv2.waitKey(1)==24:
     break
capture.release()
cv2.destroyAllWindows()
