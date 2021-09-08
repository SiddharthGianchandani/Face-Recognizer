import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import cvlib as cv
import cv2
import face_recognition 
import random
import tflearn
import numpy as np
from tkinter import *
import tkinter as tk
from tkinter import ttk 
from random import shuffle
from tqdm import tqdm
import sklearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

IMG_SIZE=100
TRAIN_DIR="authencity"
LR=1e-5
MODEL_NAME='Autenticate-{}-{}.model'.format(LR,'2conv-basic-video')
model_path="face_reco.clf"
with open(model_path, 'rb') as f:
      knn_clf = pickle.load(f)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#Readying the dataset:
def popup(img):
   nfont=("Verdana",10)
   pop=Tk()
   pop.title("Help us improve")
   pop.geometry("400x100+642+120")
   label=ttk.Label(pop, text="Please state the name of the unknown person:", font=nfont)
   label.pack()
   tf=Entry(pop)
   tf.pack()
   tf.focus_set()

   def gettext():
      fol=tf.get()
      fol=fol[:1].upper()+fol[1:].lower()
      path="train"
      fol=path+"\\"+fol
      if not os.path.exists(fol):
          os.mkdir(fol)
          #print('Folder created successfully')
          
      r=str(random.randint(0,100))
      name=r+".jpeg"
      while True:
         if name not in os.listdir(fol):
            cv2.imwrite(os.path.join(fol,name),img)
            break
         else:
            r=str(random.randint(0,100))
            name=name[:-5]+r+".jpeg"
         
      pop.destroy()
      
   B1=ttk.Button(pop,text="OK", width=10,command=gettext)
   B1.pack()
   B2=ttk.Button(pop,text="Cancel", width=10,command=pop.destroy)
   B2.pack()
   pop.mainloop()

def consider1(loc,loc1,bottom_limit,right_limit):
   for m1 in loc1:
      temp=[]
      for m in loc:
         if (m[0]-30<=m1[0]<=m[0]+30)or(m[1]-30<=m1[1]<=m[1]+30)or(m[2]-30<=m1[2]<=m[2]+30)or(m[3]-30<=m1[3]<=m[3]+30):
           continue
         temp.append(m)
      loc=temp

   for m in loc:
      loc1.append(m)
   l=[]
   for (top,right,bottom,left) in loc1:
      if bottom<=bottom_limit and right<=right_limit:
         l.append((top,right,bottom,left))
         
   return l

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

def decide(name):
    if name=="real":
        return [1,0]
    elif name=="fake":
        return [0,1]

def train_real():
    train=[]
    for fol in os.listdir(TRAIN_DIR):
        temp=decide(fol)
        fol=TRAIN_DIR+"\\"+fol
        for image in os.listdir(fol):
            path=os.path.join(fol,image)
            img=cv2.resize(cv2.imread(path),(IMG_SIZE,IMG_SIZE))
            train.append([np.array(img),np.array(temp)])
    shuffle(train)
    np.save('real_train.npy',train)
    return

def train(train_dir, model_save_path, n_neighbors, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    for fol in os.listdir(train_dir):
        temp=fol
        fol=train_dir+'\\'+fol
        for image in os.listdir(fol):
            path=os.path.join(fol,image)
            img=cv2.imread(path)
            bottom_limit=img.shape[0]
            right_limit=img.shape[1]
            loc = face_recognition.face_locations(img)
            loc1=cv.detect_face(img)[0]

            l=[]
            for m in loc1:
               m=m[1:]+m[:1]
               m=tuple(m)
               l.append(m)
            loc1=l

            loc1=consider1(loc,loc1,bottom_limit,right_limit)
            
            if len(loc1) > 0:
                X.append(face_recognition.face_encodings(img, known_face_locations=loc1)[0])
                y.append(temp)
        
    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf

'''
c=input('Do you want to train the model?(y/n):')
if c == 'y':
   classifier = train("train", model_save_path="face_reco.clf", n_neighbors=2)
'''
flag=0
#train_real()
train_data=np.load('real_train.npy',allow_pickle=True)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#Constructing the neural network

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.6)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded')





#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#Training and saving the model



train=train_data[:-35]
test=train_data[-35:]

X=np.array([i[0] for i in train])
Y=[i[1] for i in train]
 
test_x=np.array([i[0] for i in test])
test_y=[i[1] for i in test]


model.fit({'input': X}, {'targets': Y}, n_epoch=500, validation_set=({'input': test_x}, {'targets': test_y}), 
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)










#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#Testing the model
'''
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
    
    if len(loc1)>0:
        for (top, right, bottom, left) in loc1:
            if top>=0 and left>=0 and bottom<=600 and right<=600:
                temporary=img1[top:bottom,left:right]
            try:
                tempo=cv2.resize(temporary,(100,100))    

                model_out=model.predict([np.array(tempo)])[0]
                max=0
                for m in model_out:
                    if m>max:
                        max=m
                answer=[]
                for m in model_out:
                    if m==max:
                        answer.append(1)
                    else:
                        answer.append(0)
                ans=''
                if answer[0]==1:
                    ans="real"
                else:
                    ans='fake'
            
                
                if ans=="real":
                    enc = face_recognition.face_encodings(img, known_face_locations=loc1)
                    closest_distances = knn_clf.kneighbors(enc, n_neighbors=1)
                    are_matches = [closest_distances[0][i][0] <= 0.4 for i in range(len(loc1))]
                    predictions = [(pred, loc1) if rec else ("Hey", loc1) for pred, loc1, rec in zip(knn_clf.predict(enc), loc1, are_matches)]
                    for name, (top, right, bottom, left) in predictions:
                        if name=='Hey'and flag==1:
                            temp=img1.copy()
                            cv2.rectangle(img,(left,top), (right, bottom),(255, 0, 0),2)
                            cv2.putText(img,name,(left-10,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                            cv2.imshow("img",img)

                            if top-50<0:
                                temp1=top
               
                            else:
                                temp1=top-50

                            if left-50<0:
                                temp2=left
               
                            else:
                                temp2=left-50

                            right_limit=img.shape[1]
                            if right+50>right_limit:
                                temp3=right_limit
               
                            else:
                                temp3=right+50

                            bottom_limit=img.shape[0]   
                            if bottom+50>bottom_limit:
                                temp4=bottom_limit
               
                            else:
                                temp4=bottom+50
               
                            temp=temp[temp1:temp4,temp2:temp3]
                            popup(temp)
                            cv2.rectangle(img,(left,top), (right, bottom),(0, 0, 255),2)
                        else:
                            cv2.rectangle(img,(left,top), (right, bottom),(0, 0, 255),2)
                            cv2.putText(img,name,(left-10,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
         

            except:
                continue
                
    else:
        cv2.putText(img,"No One Detected",(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow("img",img)
    flag=1
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
'''
