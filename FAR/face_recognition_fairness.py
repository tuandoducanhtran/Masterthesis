## similar to the other fairness definitions with adjusted labels for merging of labels

from csv import reader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, multilabel_confusion_matrix
import numpy as np
import csv
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# intiate empty lists
fairface_label_output = []
output_deepface = []
jpg_names = []
jpg_names_index = []
y_label_age = []
y_label_gender = []
y_label_race = []
y_pred_age = []
y_pred_gender = []
y_pred_race = []
fairness_output_age = [["ACC_age", "TPR_age", "TNR_age", "FPR_age", "FNR_age", "NPP_age", "PP_age", "PPP_age"]]
fairness_output_gender = [["ACC_gender", "TPR_gender", "TNR_gender", "FPR_gender", "FNR_gender", "NPP_gender", "PP_gender", "PPP_gender"]]
fairness_output_race = [["ACC_race", "TPR_race", "TNR_race", "FPR_race", "FNR_race", "NPP_race", "PP_race", "PPP_race"]]
ACC_output_age = []
TPR_output_age = []
TNR_output_age = []
FPR_output_age = []
FNR_output_age = []
NPP_output_age = []
PP_output_age = []
PPP_output_age = []
ACC_output_gender = []
TPR_output_gender = []
TNR_output_gender = []
FPR_output_gender = []
FNR_output_gender = []
NPP_output_gender = []
PP_output_gender = []
PPP_output_gender = []
ACC_output_race = []
TPR_output_race = []
TNR_output_race = []
FPR_output_race = []
FNR_output_race = []
NPP_output_race = []
PP_output_race = []
PPP_output_race = []


# open labels from csv and save as list fairface_label_all
with open (r"C:\Users\tuanm\Desktop\MasterThesis\Dataset\Evaluation\Labels\facearg_label_eval_final_far.csv") as label_train_csv:
        csv_reader = reader(label_train_csv)
        fairface_label_all = list(csv_reader)
        
# open output from csv as list output_deepface
# CHANGE
with open (r"C:\Users\tuanm\Desktop\MasterThesis\Output\Face_recognition_master\on_facearg\output_face_recognition_on_facearg_final.csv") as output_deepface_csv:
        csv_reader = reader(output_deepface_csv)
        output_deepface = list(csv_reader)

# save element names in list
for elements in output_deepface:
    jpg_names.append(elements[0])

# map name with index    
for eintrag in jpg_names:
    jpg_names_index.append((int(eintrag.split(".jpg", 1)[0])))

for index in jpg_names_index:
    fairface_label_output.append(fairface_label_all[index])

# label in matching order with output    
for lbl in fairface_label_output:
    y_label_age.append(lbl[1])
    y_label_gender.append(lbl[2])
    y_label_race.append(lbl[3])
    
# standardize output classes for further evaluation
#y_label_age = [elements.replace("0-2", "below 20") for elements in y_label_age]
#y_label_age = [elements.replace("3-9", "below 20") for elements in y_label_age]
#y_label_age = [elements.replace("10-19", "below 20") for elements in y_label_age]
y_label_age = [elements.replace("20-29", "20-29") for elements in y_label_age]
y_label_age = [elements.replace("30-39", "30-39") for elements in y_label_age]
y_label_age = [elements.replace("40-49", "40-49") for elements in y_label_age]
y_label_age = [elements.replace("50-59", "50+") for elements in y_label_age]
y_label_age = [elements.replace("60-69", "50+") for elements in y_label_age]
y_label_age = [elements.replace("more than 70", "50+") for elements in y_label_age]

#y_label_gender = [elements.replace("60-69", "60-69") for elements in y_label_gender]
#y_label_gender = [elements.replace("60-69", "60-69") for elements in y_label_gender]

y_label_race = [elements.replace("East Asian", "Asian") for elements in y_label_race]
y_label_race = [elements.replace("Indian", "Indian") for elements in y_label_race]
y_label_race = [elements.replace("Black", "Black") for elements in y_label_race]
y_label_race = [elements.replace("White", "White") for elements in y_label_race]
y_label_race = [elements.replace("Middle Eastern", "Middle Eastern") for elements in y_label_race]
y_label_race = [elements.replace("Latino Hispanic", "Latino Hispanic") for elements in y_label_race]
y_label_race = [elements.replace("Southeast Asian", "Asian") for elements in y_label_race]

# save the pred in a seperate list called y_pred_race    
for lbls in output_deepface:
    y_pred_age.append(lbls[1])
    y_pred_gender.append(lbls[2])
    y_pred_race.append(lbls[3])
    
print("The accuracy of race tested on 1000", accuracy_score(y_label_race, y_pred_race, normalize=True))
print(len(y_label_race))
print(len(y_pred_race))

# print confusion matrix
print(confusion_matrix(y_label_age, y_pred_age))
print(confusion_matrix(y_label_gender, y_pred_gender))
print(confusion_matrix(y_label_race, y_pred_race))
print(classification_report(y_label_age, y_pred_age))
print(classification_report(y_label_gender, y_pred_gender))
print(classification_report(y_label_race, y_pred_race))

# visualization of multiclass confusion matrix race
# cf_matrix_race = pd.DataFrame(confusion_matrix(y_label_race, y_pred_race), index=["Asian", "Black", "Indian", "Latino Hispanic", "Middle Eastern", "White"], columns=["Asian", "Black", "Indian", "Latino Hispanic", "Middle Eastern", "White"])
# ax = sns.heatmap(cf_matrix_race, annot=True, cmap='Blues', annot_kws={"size":16}, fmt='g')
# ax.set_title('Confusion matrix of label race on Fairface dataset\n\n');
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values');
# plt.show()

# visualization of multiclass confusion matrix age
# cf_matrix_age = pd.DataFrame(confusion_matrix(y_label_age, y_pred_age), index=["0-2", "10-19", "20-29", "3-9", "30-39", "40-49", "50-59", "60-69", "more than 70"], columns=["0-2", "10-19", "20-29", "3-9", "30-39", "40-49", "50-59", "60-69", "more than 70"])
# ax = sns.heatmap(cf_matrix_age, annot=True, cmap='Blues', annot_kws={"size":16}, fmt='g')
# ax.set_title('Confusion matrix of label age on Fairface dataset\n\n');
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values');
# plt.show()

# visualization of multiclass confusion matrix gender
# cf_matrix_gender = pd.DataFrame(confusion_matrix(y_label_gender, y_pred_gender), index=["Female", "Male"], columns=["Female", "Male"])
# ax = sns.heatmap(cf_matrix_gender, annot=True, cmap='Blues', annot_kws={"size":16}, fmt='g')
# ax.set_title('Confusion matrix of label gender on Fairface dataset\n\n');
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values');
# plt.show()

# extract class names for age, gender and race
report_age = classification_report(y_label_age, y_pred_age, output_dict=True)
dfReport_age = pandas.DataFrame(report_age).transpose()
myList_age = dfReport_age.index.tolist()
myList_age.remove("accuracy")
myList_age.remove("macro avg")
myList_age.remove("weighted avg")
fairness_output_age.append(myList_age)

report_gender = classification_report(y_label_gender, y_pred_gender, output_dict=True)
dfReport_gender = pandas.DataFrame(report_gender).transpose()
myList_gender = dfReport_gender.index.tolist()
myList_gender.remove("accuracy")
myList_gender.remove("macro avg")
myList_gender.remove("weighted avg")
fairness_output_gender.append(myList_gender)

report_race = classification_report(y_label_race, y_pred_race, output_dict=True)
dfReport_race = pandas.DataFrame(report_race).transpose()
myList_race = dfReport_race.index.tolist()
myList_race.remove("accuracy")
myList_race.remove("macro avg")
myList_race.remove("weighted avg")
fairness_output_race.append(myList_race)

confusion_class_matrix_age = multilabel_confusion_matrix(y_label_age, y_pred_age)
confusion_class_matrix_gender = multilabel_confusion_matrix(y_label_gender, y_pred_gender)
confusion_class_matrix_race = multilabel_confusion_matrix(y_label_race, y_pred_race)

for confusion_matrix in confusion_class_matrix_age:
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
    fairness_output_age.append(confusion_output)
    
    # save fairness definitions in seperate list for var
    ACC_output_age.append(ACC)
    TPR_output_age.append(TPR)
    TNR_output_age.append(TNR)
    FPR_output_age.append(FPR)
    FNR_output_age.append(FNR)
    NPP_output_age.append(NPP)
    PP_output_age.append(PP)
    PPP_output_age.append(PPP)
    
    # plot binary confusion matrix for label age
    # cf_matrix_age_class = pd.DataFrame(confusion_matrix, index = ["Unprivileged class", "Privileged class"], columns = ["Unprivileged class class", "Privileged class"])
    # ax = sns.heatmap(cf_matrix_age_class, annot=True, cmap='Blues', annot_kws={"size":16}, fmt='g')
    # ax.set_title('Binary confusion matrix of single age group on Fairface dataset\n\n');
    # ax.set_xlabel('\nPredicted Values')
    # ax.set_ylabel('Actual Values');
    # plt.show()

for confusion_matrix in confusion_class_matrix_gender:
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
    fairness_output_gender.append(confusion_output)
    
    # save fairness definitions in seperate list for var
    ACC_output_gender.append(ACC)
    TPR_output_gender.append(TPR)
    TNR_output_gender.append(TNR)
    FPR_output_gender.append(FPR)
    FNR_output_gender.append(FNR)
    NPP_output_gender.append(NPP)
    PP_output_gender.append(PP)
    PPP_output_gender.append(PPP)

    # cf_matrix_gender_class = pd.DataFrame(confusion_matrix, index = ["Unprivileged class", "Privileged class"], columns = ["Unprivileged class class", "Privileged class"])
    # ax = sns.heatmap(cf_matrix_gender_class, annot=True, cmap='Blues', annot_kws={"size":16}, fmt='g')
    # ax.set_title('Binary confusion matrix of single gender on Fairface dataset\n\n');
    # ax.set_xlabel('\nPredicted Values')
    # ax.set_ylabel('Actual Values');
    # plt.show()


for confusion_matrix in confusion_class_matrix_race:
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
    fairness_output_race.append(confusion_output)
    
    # save fairness definitions in seperate list for var
    ACC_output_race.append(ACC)
    TPR_output_race.append(TPR)
    TNR_output_race.append(TNR)
    FPR_output_race.append(FPR)
    FNR_output_race.append(FNR)
    NPP_output_race.append(NPP)
    PP_output_race.append(PP)
    PPP_output_race.append(PPP)
    
    # binary confusion matrix for race
    # cf_matrix_race_class = pd.DataFrame(confusion_matrix, index = ["Unprivileged class", "Privileged class"], columns = ["Unprivileged class class", "Privileged class"])
    # ax = sns.heatmap(cf_matrix_race_class, annot=True, cmap='Blues', annot_kws={"size":16}, fmt='g')
    # ax.set_title('Binary confusion matrix of single race on Fairface dataset\n\n');
    # ax.set_xlabel('\nPredicted Values')
    # ax.set_ylabel('Actual Values');
    # plt.show()

# calculate var    
ACC_var_age = np.var(ACC_output_age)
TPR_var_age = np.var(TPR_output_age)
TNR_var_age = np.var(TNR_output_age)
FPR_var_age = np.var(FPR_output_age)
FNR_var_age = np.var(FNR_output_age)
NPP_var_age = np.var(NPP_output_age)
PP_var_age = np.var(PP_output_age)
PPP_var_age = np.var(PPP_output_age)

ACC_var_gender = np.var(ACC_output_gender)
TPR_var_gender = np.var(TPR_output_gender)
TNR_var_gender = np.var(TNR_output_gender)
FPR_var_gender = np.var(FPR_output_gender)
FNR_var_gender = np.var(FNR_output_gender)
NPP_var_gender = np.var(NPP_output_gender)
PP_var_gender = np.var(PP_output_gender)
PPP_var_gender = np.var(PPP_output_gender)

ACC_var_race = np.var(ACC_output_race)
TPR_var_race = np.var(TPR_output_race)
TNR_var_race = np.var(TNR_output_race)
FPR_var_race = np.var(FPR_output_race)
FNR_var_race = np.var(FNR_output_race)
NPP_var_race = np.var(NPP_output_race)
PP_var_race = np.var(PP_output_race)
PPP_var_race = np.var(PPP_output_race)


output_var_age = [["ACC_var_age", "TPR_var_age", "TNR_var_age", "FPR_var_age", "FNR_var_age", "NPP_var_age", "PP_var_age", "PPP_var_age"]]
output_var_gender = [["ACC_var_gender", "TPR_var_gender", "TNR_var_gender", "FPR_var_gender", "FNR_var_gender", "NPP_var_gender", "PP_var_gender", "PPP_var_gender"]]
output_var_race = [["ACC_var_race", "TPR_var_race", "TNR_var_race", "FPR_var_race", "FNR_var_race", "NPP_var_race", "PP_var_race", "PPP_var_race"]]
var_age = (ACC_var_age, TPR_var_age, TNR_var_age, FPR_var_age, FNR_var_age, NPP_var_age, PP_var_age, PPP_var_age)
var_gender = (ACC_var_gender, TPR_var_gender, TNR_var_gender, FPR_var_gender, FNR_var_gender, NPP_var_gender, PP_var_gender, PPP_var_gender)
var_race = (ACC_var_race, TPR_var_race, TNR_var_race, FPR_var_race, FNR_var_race, NPP_var_race, PP_var_race, PPP_var_race)
list(var_age)
list(var_gender)
list(var_race)
output_var_age.append(var_age)
output_var_gender.append(var_gender)
output_var_race.append(var_race)

# f = open (r"C:\Users\tuanm\Desktop\MasterThesis\Output\Face_recognition_master\fairness_far_on_facearg_final.csv", "w", newline ="")
# writer = csv.writer(f)
# for zeile in output_var_age:
#     writer.writerow(zeile)
# for zeile4 in fairness_output_age:
#     writer.writerow(zeile4)
# for zeile2 in output_var_gender:
#     writer.writerow(zeile2)
# for zeile5 in fairness_output_gender:
#     writer.writerow(zeile5)
# for zeile3 in output_var_race:
#     writer.writerow(zeile3)
# for zeile6 in fairness_output_race:
#     writer.writerow(zeile6)
# f.close()


