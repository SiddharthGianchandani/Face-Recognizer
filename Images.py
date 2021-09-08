import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import cvlib as cv
import cv2
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','JPG','JPEG','PNG'}


def consider(loc,loc1,bottom_limit,right_limit):
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

            loc1=consider(loc,loc1,bottom_limit,right_limit)
            print(temp)
            print(loc1)
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

def predict(path, knn_clf=None, model_path=None, distance_threshold=0.5):
    
   with open(model_path, 'rb') as f:
      knn_clf = pickle.load(f)

   # Load image file and find face locations
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
   
   loc1=consider(loc,loc1,bottom_limit,right_limit)
   # If no faces are found in the image, return an empty result.
   if len(loc1) == 0:
        return []
   # Find encodings for faces in the test image
   enc = face_recognition.face_encodings(img, known_face_locations=loc1)

   # Use the KNN model to find the best matches for the test face
   closest_distances = knn_clf.kneighbors(enc, n_neighbors=1)
   are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(loc1))]

   # Predict classes and remove classifications that aren't within the threshold
   return [(pred, loc1) if rec else ("unknown", loc1) for pred, loc1, rec in zip(knn_clf.predict(enc), loc1, are_matches)]


# In[89]:


def show_pred(img_path, predictions):
    
    img=cv2.imread(img_path)
    for name, (top, right, bottom, left) in predictions:
        cv2.rectangle(img,(left,top), (right, bottom),(0, 0, 255),2)
        cv2.putText(img,name,(left-10,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

    cv2.imshow('img',img)
    while True:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


# In[90]:


classifier = train("train", model_save_path="face_reco.clf", n_neighbors=2)

for image_file in os.listdir("test"):
    full_file_path = os.path.join("test", image_file)

    predictions = predict(full_file_path, model_path="face_reco.clf")
    print(predictions)
    for name, (top, right, bottom, left) in predictions:
        print("- Found {} at ({}, {})".format(name, left, top))

    show_pred(full_file_path, predictions)

