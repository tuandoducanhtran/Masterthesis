# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 17:02:28 2022

@author: tuanm
"""

import numpy as np
import dlib
import cv2
import my_model
import glob
import os
import csv

data = []
output_agr = []
race_dict={'White':0 , 'Black':1 , 'Asian':2 , 'Indian':3 , 'Others':4}
race_inv_dict = {i:v for v , i in zip(race_dict.keys() , race_dict.values())}

gender_dict={'Male': 0 , 'Female': 1}
gender_inv_dict = {i:v for v , i in zip(gender_dict.keys() , gender_dict.values())}

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

detector = dlib.get_frontal_face_detector()

image_files = [f for f in glob.glob(r"C:\Users\tuanm\Desktop\MasterThesis\Dataset\Evaluation\Fairface\full\*", recursive = True) if not os.path.isdir(f)]
for filename in image_files:
    data.append(os.path.basename(filename))

model = my_model.My_Model(weights_path = r'C:\Users\tuanm\Desktop\MasterThesis\Models\AGR_ft\finetune_fairface\utk_newage_finetuned.h5')
#model = my_model.My_Model(weights_path = r'C:\Users\tuanm\Desktop\MasterThesis\Models\AGR\recognition_age_3.h5')

for imgs in image_files:
    try:
        picturename = os.path.basename(imgs)
        img = cv2.imread(imgs)
        #img = cv2.imread(r'C:\Users\tuanm\OneDrive\Desktop\MasterThesis\Age-Gender-Race-recognition-master\3.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rects = detector(gray , 1)
        margin = 10
        faces = []
        if len(rects)>0:
        	# for  i in range(len(rects)):
        	(x,y,w,h) = rect_to_bb(rects[0])
        	cv2.rectangle(img,(x-margin,y-margin),(x+w+margin,y+h+margin),(255,0,0),2)
        	face = cv2.resize( img[y-margin:y+h+margin, x-margin:x+w+margin], (64 , 64))
        	faces.append(face)        		
        faces = np.array(faces)
        g , a ,r = model.predict(faces)
        a = a.T
        g = g.T
        r = r.T
        age = np.argmax(a)
        gender = np.argmax(g)
        race = np.argmax(r)
        agr = picturename, age , gender_inv_dict[gender]  , race_inv_dict[race]
        values = list(agr)
        print(values)
        output_agr.append(values)
    except Exception:
        none = picturename, 'none', 'none', 'none'
        values = list(none)
        output_agr.append(values)
        print(values)

for values in output_agr:
    if values[1] != "none":
        int(values[1])
        if values[1] == 0:
            values[1] = '0-2'
        elif values[1] == 1:
            values[1] = '3-9'
        elif values[1] == 2:
            values[1] = '10-19'
        elif values[1] == 3:
            values[1] = '20-29'
        elif values[1] == 4:
            values[1] = '30-39'
        elif values[1] == 5:
            values[1] = '40-49'
        elif values[1] == 6:
            values[1] = '50-59'
        elif values[1] == 7:
            values[1] = '60-69'
        elif values[1] == 8:
            values[1] = 'more than 70'
    
    
        
f = open (r"C:\Users\tuanm\Desktop\MasterThesis\Output\AGR_ft\on_fairface\output_agrnewft1_on_fairface.csv", "w", newline ="")
writer = csv.writer(f)
for zeile in output_agr:
    writer.writerow(zeile)
f.close()

print("output_agr has been saved.")