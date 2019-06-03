# -*- coding: utf-8 -*-
''' face recognition'''
import cv2
import os
from pynput.mouse import Listener,Controller,Button
from collections import deque
#import time

speed_x=30
speed_y=30
n=10
path = os.getcwd()
face_cascade = cv2.CascadeClassifier(path+'/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

face_centers1=deque()
face_centers2=deque()
#fps of this program is 8.55
click_queue=deque()
click_frame=15 
click_pixels=3

mouse=Controller()

def mean(face_centers,n):
  fcx=int(sum([i[0] for i in face_centers ])/n)
  fcy=int(sum([i[1] for i in face_centers ])/n)
  return (fcx,fcy)
    
def distance(face_centers1,face_centers2,n):
  (fcx1,fcy1)=mean(face_centers1,n)
  (fcx2,fcy2)=mean(face_centers2,n)
  (rel_x,rel_y) = fcx2-fcx1,fcy2-fcy1
  return (rel_x,rel_y)

def on_move(x, y):
  event.append(1)
  #print("on_move")
  

def on_click(x, y, button, pressed):
  event.append(1)
  click_queue.popleft()
  click_queue.append(1)
  #print("On_click")
 
    

def on_scroll(x, y, dx, dy):
  event.append(1)
  #print("On_scroll")
  
#fps=0
#time_start = time.time()

while True:
  event= list([])  
  listener= Listener(on_move=on_move, on_click = on_click, on_scroll = on_scroll) 
  listener.start()

  ret,img = cap.read()
  img=cv2.resize(img,(640,480))
  img=cv2.flip(img,1)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray,1.3,5)
  
  for (x,y,w,h) in faces:
    center=([cx,cy])=(x+w/2),(y+h/2)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("face",img)
    
    if len(face_centers2)==n:
      face_centers1=face_centers2.copy()
      ret=face_centers2.popleft()
      face_centers2.append(center)
      (rel_x,rel_y)=distance(face_centers1,face_centers2,n)
      if len(event) == 0:
         #print("rel_x",rel_x,"   rel_y",rel_y)
         mouse.move( speed_x*rel_x, speed_y*rel_y)
         if (abs(rel_x)<click_pixels and abs(rel_y)<click_pixels):    #no. of pixels for uncertaintity in stable position
            if len(click_queue)<click_frame: 
               click_queue.append(0)
               #print("less than click frame",click_queue)
            else:
               click_queue.popleft()
               click_queue.append(0)
               #print("more than click frame",click_queue)
         else:
            if len(click_queue)<click_frame: 
               click_queue.append(1)
               #print("not abs",click_queue)
            else:
               click_queue.popleft()
               click_queue.append(1)
      #else:
        #print("at rest")
    else: 
      face_centers2.append(center)
      
    if len(click_queue)==click_frame and all([i==0 for i in click_queue]):
      mouse.press(Button.left)
      mouse.release(Button.left)
      #print(click_queue)
      
  #fps+=1    
  listener.stop()  
  key = cv2.waitKey(50) & 0xff
  if key == ord("q"):
      break

#fps = fps/ (time.time() - time_start)
#print("fps is ",fps)
cap.release()
cv2.destroyAllWindows()  

