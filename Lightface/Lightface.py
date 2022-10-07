
"""
Created on Tue Aug  2 18:48:13 2022
# source: https://github.com/serengil/deepface
# this code imports lightface/Deepface and performs the prediction on the given folder stated as image_files
# the pictures in the folder are then predicted, the labels are saved as .csv

@author: tuanm
"""

from deepface import DeepFace
from csv import reader
import csv
import os.path
import glob


# define keys for map values
keys_deepface = ["age", "gender", "dominant_race"]
data = []
fairface_label = []

# open csv for to write in output
f = open (r"C:\Users\tuanm\Desktop\MasterThesis\Output\Deepface\output_deepface.csv", "w", newline ="")
writer = csv.writer(f)
counter = 0

# read image names in folder
image_files = [f for f in glob.glob(r"C:\Users\tuanm\Desktop\MasterThesis\Dataset\Evaluation\FaceARG\*", recursive = True) if not os.path.isdir(f)]
# image_files = [f for f in glob.glob(r"C:\Users\tuanm\Desktop\MasterThesis\Dataset\Fairface\crop\eval\1\*", recursive = True) if not os.path.isdir(f)]
for filename in image_files:
    data.append(os.path.basename(filename))    
    
# for every element in list use deepface and save result in output.csv
for filename in data:
    try:
        file_path = "C:/Users/tuanm/Desktop/MasterThesis/Dataset/Evaluation/FaceARG/" + filename
        # file_path = "C:/Users/tuanm/Desktop/MasterThesis/Dataset/Fairface/crop/eval/1/" + filename
        result_set = DeepFace.analyze(img_path = file_path)
        values = list(map(result_set.get, keys_deepface))
        values.insert(0, filename)
        # map the output in age-ranges (like fairface dataset) for further comparison
        if 0 <= values[1] <=2:
            values[1] = '0-2'
        elif 3 <= values[1] <=9:
            values[1] = '3-9'
        elif 10 <= values[1] <=19:
            values[1] = '10-19'
        elif 20 <= values[1] <=29:
            values[1] = '20-29'
        elif 30 <= values[1] <=39:
            values[1] = '30-39'
        elif 40 <= values[1] <=49:
            values[1] = '40-49'
        elif 50 <= values[1] <=59:
            values[1] = '50-59'
        elif 60 <= values[1] <=69:
            values[1] = '60-69'
        elif 70 <= values[1]:
            values[1] = 'more than 70'
            
        # change output for mapping with fairface attributes          
        new_values = [elements.replace("Man", "Male") for elements in values]
        new_values = [elements.replace("Woman", "Female") for elements in new_values]
        new_values = [elements.replace("indian", "Indian") for elements in new_values]
        new_values = [elements.replace("black", "Black") for elements in new_values]
        new_values = [elements.replace("white", "White") for elements in new_values]
        new_values = [elements.replace("middle eastern", "Indian") for elements in new_values]
        new_values = [elements.replace("latino hispanic", "Indian") for elements in new_values]
        new_values = [elements.replace("asian", "Asian") for elements in new_values]
        
        # write output in csv        
        writer.writerow(new_values)
        counter = counter + 1
        print(str(counter) + "/" + str(len(data)))
        
        # exception if no face was found
    except Exception:
        none = filename, 'none', 'none', 'none'
        values = list(none)
        writer.writerow(values)
        counter = counter + 1
        print(str(counter) + "/" + str(len(data)))
        
f.close()


# used to save results in .csv 
# with open (r"C:\Users\tuanm\Desktop\MasterThesis\Output\Deepface\output_deepface_on_Fairface_cropped.csv") as output_deepface_csv:
#         csv_reader = reader(output_deepface_csv)
#         output_deepface = list(csv_reader)

