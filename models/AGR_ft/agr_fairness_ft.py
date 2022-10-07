# -*- coding: utf-8 -*-
# source: https://github.com/kuabhish/Age-Gender-Race-recognition
"""
Created on Sun Sep  4 17:33:45 2022

@author: tuanm
"""

from csv import reader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, multilabel_confusion_matrix
import numpy as np
import csv

# intiate empty lists
fairface_label_output = []
output_deepface = []
jpg_names = []
jpg_names_index = []
y_label_race = []
y_pred_race = []
fairness_output = [["ACC", "TPR", "TNR", "FPR", "FNR", "NPP", "PP", "PPP"]]
ACC_output = []
TPR_output = []
TNR_output = []
FPR_output = []
FNR_output = []
NPP_output = []
PP_output = []
PPP_output = []


# open labels from csv and save as list fairface_label_all
with open (r"C:\Users\tuanm\Desktop\MasterThesis\Dataset\Fairface\fairface_label_train.csv") as label_train_csv:
        csv_reader = reader(label_train_csv)
        fairface_label_all = list(csv_reader)
        
# open output from csv as list output_deepface
with open (r"C:\Users\tuanm\Desktop\MasterThesis\Output\AGR_ft\output_agr_ft.csv") as output_agr_csv:
        csv_reader = reader(output_agr_csv)
        output_agr = list(csv_reader)

# save element names in list
for elements in output_agr:
    jpg_names.append(elements[0])

# map name with index    
for eintrag in jpg_names:
    jpg_names_index.append((int(eintrag.split(".jpg", 1)[0])))

for index in jpg_names_index:
    fairface_label_output.append(fairface_label_all[index])

# label in matching order with output    
for lbl in fairface_label_output:
    y_label_race.append(lbl[3])
    
# standardize output classes
y_label_race = [elements.replace("East Asian", "Asian") for elements in y_label_race]
y_label_race = [elements.replace("Indian", "Indian") for elements in y_label_race]
y_label_race = [elements.replace("Black", "Black") for elements in y_label_race]
y_label_race = [elements.replace("White", "White") for elements in y_label_race]
y_label_race = [elements.replace("Middle Eastern", "Others") for elements in y_label_race]
y_label_race = [elements.replace("Latino_Hispanic", "Others") for elements in y_label_race]
y_label_race = [elements.replace("Southeast Asian", "Asian") for elements in y_label_race]

# save the pred in a seperate list called y_pred_race    
for lbls in output_agr:
    y_pred_race.append(lbls[3])
    
print("The accuracy of race tested on image 1.jpg - 1000.jpg is:", accuracy_score(y_label_race, y_pred_race, normalize=True))
print(len(y_label_race))
print(len(y_pred_race))

print(confusion_matrix(y_label_race, y_pred_race))
print(classification_report(y_label_race, y_pred_race))

confusion_class_matrix = multilabel_confusion_matrix(y_label_race, y_pred_race)
confusion_matrix_asian = confusion_class_matrix[0]
confusion_matrix_black = confusion_class_matrix[1]
confusion_matrix_indian = confusion_class_matrix[2]
confusion_matrix_latino = confusion_class_matrix[3]
confusion_matrix_middle = confusion_class_matrix[4]
confusion_matrix_white = confusion_class_matrix[5]

for confusion_matrix in confusion_class_matrix:
    confusion_output = []
    TN = confusion_matrix[0][0]
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    
    # fairness definitions
    # sum of confusion matrix
    N = TN + TP + FP + FN
    # accuracy/overall accuracy equality
    ACC = (TP+TN)/N
    # true positive rate/equal opportunity
    TPR = TP/(TP+FN)
    # true negative rate
    TNR = TN/(TN+FP) 
    # false positive rate/predictive equality
    FPR = FP/(FP+TN)
    # false negative rate/treatment equality
    FNR = FN/(TP+FN)
    # negative predictive parity
    NPP = TN/(TN+FN)
    # predictive parity
    PP = TP/(TP+FP)
    # percentage predicted as positive/statistical parity
    PPP = (TP+FP)/N
    
    # prepare fairness output
    confusion_output.append(ACC)
    confusion_output.append(TPR)
    confusion_output.append(TNR)
    confusion_output.append(FPR)
    confusion_output.append(FNR)
    confusion_output.append(NPP)
    confusion_output.append(PP)
    confusion_output.append(PPP)    
    fairness_output.append(confusion_output)
    
    # save fairness definitions in seperate list for var
    ACC_output.append(ACC)
    TPR_output.append(TPR)
    TNR_output.append(TNR)
    FPR_output.append(FPR)
    FNR_output.append(FNR)
    NPP_output.append(NPP)
    PP_output.append(PP)
    PPP_output.append(PPP)
    
# calculate var    
ACC_var = np.var(ACC_output)
TPR_var = np.var(TPR_output)
TNR_var = np.var(TNR_output)
FPR_var = np.var(FPR_output)
FNR_var = np.var(FNR_output)
NPP_var = np.var(NPP_output)
PP_var = np.var(PP_output)
PPP_var = np.var(PPP_output)

output_var = [["ACC_var", "TPR_var", "TNR_var", "FPR_var", "FNR_var", "NPP_var", "PP_var", "PPP_var"]]
var = (ACC_var, TPR_var, TNR_var, FPR_var, FNR_var, NPP_var, PP_var, PPP_var)
list(var)
output_var.append(var)

f = open (r"C:\Users\tuanm\Desktop\MasterThesis\Output\AGR_ft\agr_ft_fairness.csv", "w", newline ="")
writer = csv.writer(f)
for zeile in output_var:
    writer.writerow(zeile)
f.close()