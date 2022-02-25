# Driver-Drowsiness-Detection-Alert-System-

Driver Drowsiness Detection Alert System with Open-CV & Keras Using IP-webCam For Camera Connection

Ajinkya KhandaveClick here to view Ajinkya Khandave’s profile
Ajinkya Khandave
𝐌𝐋𝐎𝐩𝐬 || 𝐃𝐞𝐯𝐎𝐩𝐬 || 𝐇𝐲𝐛𝐫𝐢𝐝 𝐚𝐧𝐝 𝐌𝐮𝐥𝐭𝐢 𝐂𝐥𝐨𝐮𝐝 𝐂𝐨𝐦𝐩𝐮𝐭𝐢𝐧𝐠 ||
Published Sep 20, 2020
+ Follow
***A countless number of people drive on the highway day and night. Taxi drivers, bus drivers, truck drivers, and people traveling long-distance suffer from lack of sleep. Due to which it becomes very dangerous to drive when feeling sleepy.

The majority of accidents happen due to the drowsiness of the driver. So, to prevent these accidents we will build a system using Python, OpenCV, and Keras which will alert the driver when he feels sleepy.

Drowsiness detection is a safety technology that can prevent accidents that are caused by drivers who fell asleep while driving.

The objective of this intermediate Machine Learning project is to build a drowsiness detection system that will detect that a person’s eyes are closed for a few seconds. This system will alert the driver when drowsiness is detected.

In this Python project, we will be using OpenCV for gathering the images from the webcam here I used IPwebcam from a mobile camera view and feed them into a Deep Learning model which will classify whether the person’s eyes are ‘Open’ or ‘Closed’. The approach we will be using for this Python project is as follows :

The required Dataset
The dataset used for this model is created by us. To create the dataset, we wrote a script that captures eyes from a camera and stores in our local disk. We separated them into their respective labels ‘Open’ or ‘Closed’. The data was manually cleaned by removing the unwanted images which were not necessary for building the model. The data comprises around 7000 images of people’s eyes under different lighting conditions. After training the model on our dataset, we have attached the final weights and model architecture file “models/cnnCat2.h5”.

The Model Architecture
The model we used is built with Keras using Convolutional Neural Networks. A convolutional neural network is a special type of deep neural network which performs extremely well for image classification purposes. A CNN basically consists of an input layer, an output layer, and a hidden layer which can have multiple numbers of layers. A convolution operation is performed on these layers using a filter that performs 2D matrix multiplication on the layer and filter.

The CNN model architecture consists of the following layers:

Convolutional layer; 32 nodes, kernel size 3
Convolutional layer; 32 nodes, kernel size 3
Convolutional layer; 64 nodes, kernel size 3
Fully connected layer; 128 nodes
The final layer is also a fully connected layer with 2 nodes.

No alt text provided for this image
Prerequisites
The requirement for this Python project is a web cam through which we will capture images. You need to have Python (3.7version recommended) installed on your system, then using pip, you can install the necessary packages.

OpenCV – pip install opencv-python (face and eye detection).
TensorFlow – pip install tensorflow (Keras uses TensorFlow as backend).
Keras – pip install keras (to build our classification model).
Pygame – pip install pygame (to play alarm sound).
No alt text provided for this image
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
The “haar cascade files” folder consists of the XML files that are needed to detect objects from the image. In our case, we are detecting the face and eyes of the person.
The models' folder contains our model file “cnnCat2.h5” which was trained on convolutional neural networks.
We have an audio clip “alarm.wav” which is played when the person is feeling drowsy.
Which is useful for the warning to the driver or RTO officers.
“Model.py” file contains the program through which we built our classification model by training on our dataset. You could see the implementation of the convolutional neural network in this file.
“Drowsiness detection.py” is the main file of our project. To start the detection procedure, we have to run this file.
Before Creating the Model we have to train the model using CNN 

For the training model, I created a model.py file that created "cnnCat2.h5" in the model folder on my computer.

Keras is used to the training of models, following library are used:-

import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model
A convolution operation is performed on these layers using a filter that performs 2D matrix multiplication on the layer and filter.

def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 32
TS=(24,24)
train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)






model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#32 convolution filters used each of size 3x3
#again
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
For randomly turn neurons on and off to improve convergence used 

  Dropout(0.25), model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS) model.save('models/cnnCat2.h5', overwrite=True)

The following model gets saved using a model.save function. Now, cnnCat2.h5 file is used to load in our model by Following steps. 

Now let's see Steps:-

Step 1 – Take Image as Input from a Camera
With a webcam, we will take images as input. So to access the webcam, we made an infinite loop that will capture each frame. We use the method provided by OpenCV, cv2.VideoCapture('http://192.168.43.1:8080/video') to access the camera and set the capture object (cap). cap.read() will read each frame and we store the image in a frame variable.

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture('http://192.168.43.1:8080/video')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
Step 2 – Detect Face in the Image and Create a Region of Interest (ROI)
To detect the face in the image, we need to first convert the image into grayscale as the OpenCV algorithm for object detection takes gray images in the input. We don’t need color information to detect the objects. We will be using a haar cascade classifier to detect faces. This line is used to set our classifier face = cv2.CascadeClassifier(‘ path to our haar cascade xml file’). Then we perform the detection using faces = face.detectMultiScale(gray). It returns an array of detections with x,y coordinates, and height, the width of the boundary box of the object. Now we can iterate over the faces and draw boundary boxes for each face.

for (x,y,w,h) in faces: 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,100,100), 1 )
Step 3 – Detect the eyes from ROI and feed it to the classifier
The same procedure to detect faces is used to detect eyes. First, we set the cascade classifier for eyes in leye and reye respectively then detect the eyes using left_eye = leye.detectMultiScale(gray). Now we need to extract only the eyes data from the full image. This can be achieved by extracting the boundary box of the eye and then we can pull out the eye image from the frame with this code.

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
For RIGHT EYE :-

for (x,y,w,h) in right_eye:

        r_eye=frame[y:y+h,x:x+w]

        count=count+1

        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)

        r_eye = cv2.resize(r_eye,(24,24))

        r_eye= r_eye/255

        r_eye=  r_eye.reshape(24,24,-1)

        r_eye = np.expand_dims(r_eye,axis=0)

        rpred = model.predict_classes(r_eye)

        if(rpred[0]==1):

            lbl='Open' 

        if(rpred[0]==0):

            lbl='Closed'

        break
For LEFT EYE :-

for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break
Step 4 – Classifier will Categorize whether Eyes are Open or Closed
We are using CNN classifier for predicting the eye status. To feed our image into the model, we need to perform certain operations because the model needs the correct dimensions to start with. First, we convert the color image into grayscale using r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY). Then, we resize the image to 24*24 pixels as our model was trained on 24*24 pixel images cv2.resize(r_eye, (24,24)). We normalize our data for better convergence r_eye = r_eye/255 (All values will be between 0-1). Expand the dimensions to feed into our classifier. We loaded our model using model = load_model(‘models/cnnCat2.h5’) . Now we predict each eye with our model

lpred = model.predict_classes(l_eye). If the value of lpred[0] = 1, it states that eyes are open, if value of lpred[0] = 0 then, it states that eyes are closed.

Step 5 – Calculate Score to Check whether Person is Drowsy
The score is basically a value we will use to determine how long the person has closed his eyes. So if both eyes are closed, we will keep on increasing the score and when eyes are open, we decrease the score. We are drawing the result on the screen using the cv2.putText() function which will display the real-time status of the person. Take a frame gives an output when the eye is open the show msg that "Open Happy journey"When eyes is closed them gives warning msg "Closed Warning".



if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed Warning",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open Happy Journey",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        ##person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('MLOps Project',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
A threshold is defined for example if the score becomes greater than 15 that means the person’s eyes are closed for a long period of time. This is when we beep the alarm using sound.play()

Demo:-
When Eyes are closed then it shows red warning with sound and msg displayed is "Closed warning"

When Eyes are open then its normally display msg "Open Happy Journey" and Display normal screen 

Summary
In this Machine Learning project, I have built a drowsy driver alert system that you can implement in numerous ways. We used OpenCV to detect faces and eyes using a haar cascade classifier and then we used a CNN model to predict the status.

If you liked the Intermediate Python Project on Drowsiness Detection System, do share and plz give me suggestions about the project and DM me if you have any queries.

GitHub Link
