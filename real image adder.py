import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import cvlib as cv
import cv2
import face_recognition 
from tkinter import *
import tkinter as tk
from tkinter import ttk 
import random


fol="authencity\\real"

def put_img(img):
    i=0
    r="0"
    name=r+".jpeg"
    while True:
        if name not in os.listdir(fol):
            cv2.imwrite(os.path.join(fol,name),img)
            break
        else:
            i+=1
            r=str(i)
            name="new"+r+".jpeg"
             
def consider(loc,loc1):
    for m1 in loc1:
        temp=[]
        for m in loc:
            if (m[0]-30<=m1[0]<=m[0]+30)or(m[1]-30<=m1[1]<=m[1]+30)or(m[2]-30<=m1[2]<=m[2]+30)or(m[3]-30<=m1[3]<=m[3]+30):
                continue
            temp.append(m)
        loc=temp

    for m in loc:
        loc1.append(m)
    return loc1

cap=cv2.VideoCapture(0)
while True:
    ret,img=cap.read()
    img1=img.copy()
    loc = face_recognition.face_locations(img)
    loc1=cv.detect_face(img)[0]
    l=[]
    for m in loc1:
        m=m[1:]+m[:1]
        m=tuple(m)
        l.append(m)
    loc1=l
    loc1=consider(loc,loc1)
    print(loc1)
    if len(loc1)>0:
        for (top, right, bottom, left) in loc1:
            cv2.rectangle(img,(left,top), (right, bottom),(255, 0, 0),2)
            temp=img1[top-50:bottom+50,left-50:right+50]
            temp=cv2.resize(temp,(100,100))
            put_img(temp)
    else:
        cv2.putText(img,"No One Detected",(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow("img",img)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
