# source: https://github.com/ChayanBansal/Face_Recognition
## we start by calling the pre trained model
## we then classify each face with cv2 and the haarcascade classifier in lin 34
## the faces are then used for age gender and racial classification
'''
This python file is a part of the proposed tentative solution to the problem statement 'Face Recognition - Age, Ethnicity and Emotion classification'.
Note: The model used for Face Recognition will be improved in the the comming version.
'''

import numpy as np
import cv2
import glob
import os
import csv
import visualkeras
from tensorflow.keras.layers import Conv2D , Dense , Flatten, MaxPooling2D , Dropout , BatchNormalization, Input

# Loading output_mapper
# This dictionary maps the predicted output with the text
import pickle
output_mapper = pickle.load(open (r"C:\Users\tuanm\Desktop\MasterThesis\Models\Face_recognition_master\output_mapper.p", "rb")) 
outputs = []
all_outputs = []
data = []
output_sorted = []
# Loading model
# A CNN model trained on the dataset provided (Face_Recognition.json)
from keras.models import load_model
model = load_model(r'C:\Users\tuanm\Desktop\MasterThesis\Models\Face_recognition_master\production_model.h5')
print(model.summary())
#visualkeras.layered_view(model, legend=True, to_file=r'C:\Users\tuanm\Desktop\MasterThesis\Models\Face_recognition_master\FAR_structure.png', type_ignore=[BatchNormalization])
IMAGE_WIDTH = IMAGE_HEIGHT = 100 # Depends on model the model used
IMAGE_CHANNELS=1 # Depends on model the model used

face_cascade = cv2.CascadeClassifier(r'C:\Users\tuanm\Desktop\MasterThesis\Models\Face_recognition_master\haarcascade_frontalface_default.xml') # Haar Cascade for face detection
########################################################################################################



# SET VALUES AS PER REQUIREMENT
#LOCATION_VIDEO = '/absolute/path/to/the/testvideo.mp4' # For real time prediction. Set to 0 (zero) if using webcamp or else the string format absolute path to the video file
#LOCATION_IMAGE = r"C:\Users\tuanm\OneDrive\Desktop\MasterThesis\Dataset\full\random_train\1.jpg" # For non real time prediction (i.e., not from live stream or video )

image_files = [f for f in glob.glob(r"C:\Users\tuanm\Desktop\MasterThesis\Dataset\Evaluation\FaceARG\*", recursive = True) if not os.path.isdir(f)]
for filename in image_files:
    data.append(os.path.basename(filename))
    
    # Comment this code block of code and uncomment the next block for Realtime (live) prediction from webcam
    ###############################################################################
for imgs in image_files:
    picturename = os.path.basename(imgs)
    LOCATION_IMAGE = imgs
    img = cv2.imread(LOCATION_IMAGE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 1)
    if len(faces)!=0:
        faces = faces[[0]]
    if len(faces)==0:
        none = picturename, 'none', 'none', 'none'
        values = list(none)
        all_outputs.append(values)
        print(none)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray,(IMAGE_HEIGHT, IMAGE_WIDTH))
        roi_gray = roi_gray / 255.0
        pred = model.predict(np.array(roi_gray).reshape(1,IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        pred_dict = dict(zip(['age', 'race', 'emotion', 'gender'], pred ))
        print('-------------Another Face-----------') # Showing output in Terminal
        text = ''
        outputz = [picturename]
        for key, value in pred_dict.items():
            text+= str(output_mapper[key][np.argmax(value[0])]) + ''
            output = (output_mapper[key][np.argmax(value[0])])
            outputz.append(output)
            print(output_mapper[key][np.argmax(value[0])]) # Showing output in Terminal
        all_outputs.append(outputz)
        cv2.putText(img,text, (x,y),cv2.FONT_HERSHEY_DUPLEX,0.5,(200,0,0),1)
        
for elements in all_outputs:
    if elements[1] != "none":
        elements.pop(3)
    
for elements in all_outputs:
    if elements[1] != "none":
        myorder = [0, 1, 3, 2]
        elements = [elements[i] for i in myorder]
        output_sorted.append(elements)
    else:
        output_sorted.append(elements)

for eintrag in output_sorted:
        if eintrag[1] == "age_20_30":
            eintrag[1] = "20-29"
        if eintrag[1] == "age_30_40":
            eintrag[1] = "30-39"
        if eintrag[1] == "age_40_50":
            eintrag[1] = "40-49"
        if eintrag[1] == "age_above_50":
            eintrag[1] = "50+"
        if eintrag[1] == "age_below20":
            eintrag[1] = "below 20"
        if eintrag[2] == "g_male":
            eintrag[2] = "Male"
        if eintrag[2] == "g_female":
            eintrag[2] = "Female"       
        if eintrag[3] == "e_arab":
            eintrag[3] = "Indian"
        if eintrag[3] == "e_asian":
            eintrag[3] = "Asian"
        if eintrag[3] == "e_black":
            eintrag[3] = "Black"
        if eintrag[3] == "e_hispanic":
            eintrag[3] = "Indian"
        if eintrag[3] == "e_indian":
            eintrag[3] = "Indian"
        if eintrag[3] == "e_white":
            eintrag[3] = "White"
            
f = open (r"C:\Users\tuanm\Desktop\MasterThesis\Output\Face_recognition_master\output_face_recognition.csv", "w", newline ="")
writer = csv.writer(f)
for zeile in output_sorted:
    writer.writerow(zeile)
f.close()
print("output_face_recognition.csv has been saved.")


    
        # new_values = [elements.replace("g_male", "Male") for elements in einträge]
        # new_values = [elements.replace("Woman", "Female") for elements in new_values]
        # new_values = [elements.replace("indian", "Indian") for elements in new_values]
        # new_values = [elements.replace("black", "Black") for elements in new_values]
        # new_values = [elements.replace("white", "White") for elements in new_values]
        # new_values = [elements.replace("middle eastern", "Middle Eastern") for elements in new_values]
        # new_values = [elements.replace("latino hispanic", "Latino Hispanic") for elements in new_values]
        # new_values = [elements.replace("asian", "Asian") for elements in new_values]    
        
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ###############################################################################



# Uncomment the following code for RealTime (live) prediction from the Webcam
###############################################################################
# cap=cv2.VideoCapture()
# while True:
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 1)
#     if len(faces)==0:
#         print('No face detected')
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#         roi_gray = cv2.resize(roi_gray,(IMAGE_HEIGHT, IMAGE_WIDTH))
#         roi_gray = roi_gray / 255.0
#         pred = model.predict(np.array(roi_gray).reshape(1,IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
#         pred_dict = dict(zip(['age', 'race', 'emotion', 'gender'], pred ))
#         print('-------------Another Face-----------') # Showing output in Terminal
#         text = ''
#         for key, value in pred_dict.items():
#             text+= str(output_mapper[key][np.argmax(value[0])]) + ''
#             print(output_mapper[key][np.argmax(value[0])]) # Showing output in Terminal
#         cv2.putText(img,text, (x,y),cv2.FONT_HERSHEY_DUPLEX,0.5,(200,0,0),1)
#     cv2.imshow('Face Recognition',img)
#     if cv2.waitKey(10) == ord('q'): # wait until 'q' key is pressed
#         break
# cap.release()
# cv2.destroyAllWindows
###############################################################################