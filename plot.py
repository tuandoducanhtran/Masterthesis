# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:48:14 2022

@author: tuanm
"""
import plotnine as p9
import pandas as pd

#pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\VAR_all.xlsx", sep='\t', lineterminator='\r', encoding = "utf-8")
df_var_all = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\VAR_all.csv")
df_detection_all = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\Detection_Rate.csv")
df_detection_cropvsfull = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\Detection_Rate_testonfairface.csv")
df_accuracy = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\Accuracy_score.csv")
df_accuracy_avg = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\AVG_acc_score.csv")
df_var_trans = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\VAR_all_trans.csv")
df_var_trans_noft = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\VAR_all_trans_noft.csv")
df_var_trans_ft = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\VAR_all_trans_noft.csv")
df_accuracy_ft = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\Accuracy_score_ft.csv")
df_var_trans_noft_avg = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\Var_all_trans_noft_avg.csv")
df_dataset = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\dataset_composition.csv")

plot_detection_all = (p9.ggplot(data=df_detection_all, mapping = p9.aes(x="model", y="detection_rate")) 
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (4, 4))
        + p9.labs(x="ML model", y= "detection rate", title=("Face detection rate of each model"))
        + p9.facet_grid(".~dataset")
        )
#plot_detection_all.save(filename=r"C:\Users\tuanm\Desktop\MasterThesis\Output\Plots\detection_rate_sw.png", height=6 , width = 10)

# plot_detection_cropvsfull = (p9.ggplot(data=df_detection_cropvsfull, mapping = p9.aes(x="model", y="detection_rate", fill="dataset"))
#         + p9.geom_col(position="dodge")
#         + p9.theme(axis_text_x = p9.element_text(angle=360), figure_size = (4, 4))
#         + p9.labs(x="fairness model", y= "detection rate", title=("Face detection rate of on Fairface dataset"))
#         + p9.theme(
#             #legend_position="bottom",
#             legend_position=(.5, -.1),
#             legend_direction="horizontal",
#             legend_title_align="center",
#         )
#         )

# df_detection = pd.read_csv(r"C:\Users\X\Desktop\THESIS\Ergebnisse\Plots\plotnine\Detection_Rate.csv")


# detection_plot = (p9.ggplot(data=df_detection, mapping = p9.aes(x="model", y="detection_rate", fill="dataset")) 
#         + p9.geom_col(position="dodge")
#         + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (4, 4))
#         + p9.labs(x="fairness model", y= "variance", title=("variance for each model"))
#         )
# detection_plot.save(filename=r"C:\Users\X\Desktop\THESIS\Ergebnisse\Plots\plotnine\detection_rate.png", height=5 , width = 10)
# #size 6, 10
# plot_detection_cropvsfull.save(filename=r"C:\Users\tuanm\Desktop\MasterThesis\Output\Plots\detection_cropvsfull.png", height=4 , width = 8)

plot_detection_cropvsfull = (p9.ggplot(data=df_detection_cropvsfull, mapping = p9.aes(x="model", y="detection_rate"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (4, 4))
        + p9.labs(x="ML model", y= "detection rate", title=("Face detection rate on Fairface dataset"))
        + p9.facet_grid(".~dataset")
        )

#size 6, 10
#plot_detection_cropvsfull.save(filename=r"C:\Users\tuanm\Desktop\MasterThesis\Output\Plots\detection_cropvsfull.png", height=6 , width = 10)

plot_accuracy = (p9.ggplot(data=df_accuracy, mapping = p9.aes(x="model", y="accuracy_score", fill="dataset"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (10, 4))
        + p9.labs(x="ML model", y= "accuracy score", title=("Accuracy for each classification on all datasets"))
        + p9.facet_grid(".~label")
        )

plot_accuracy_avg = (p9.ggplot(data=df_accuracy_avg, mapping = p9.aes(x="model", y="avg acc"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (10, 4))
        + p9.labs(x="ML model", y= "accuracy score", title=("Average accuracy of all models"))
        + p9.facet_grid(".~label")
        )

plot_accuracy_ft = (p9.ggplot(data=df_accuracy_ft, mapping = p9.aes(x="model", y="accuracy_score"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (10, 4))
        + p9.labs(x="ML model", y= "accuracy score", title=("Accuracy for each finetuned model on Fairface dataset"))
        + p9.facet_grid(".~label")
        )

## sub data frames for each dataset for further plotting
df_var_facearg = df_var_trans.loc[(df_var_trans["dataset"] == "FaceARG"),
                                  ["model", "label", "fairness_def", "var"]]

df_var_facearg_age = df_var_facearg.loc[(df_var_facearg["label"] == "age"),
                                  ["model", "fairness_def", "var"]]
df_var_facearg_gender = df_var_facearg.loc[(df_var_facearg["label"] == "gender"),
                                  ["model", "fairness_def", "var"]]
df_var_facearg_race = df_var_facearg.loc[(df_var_facearg["label"] == "race"),
                                  ["model", "fairness_def", "var"]]


##
df_var_fairface = df_var_trans_noft.loc[(df_var_trans_noft["dataset"] == "Fairface"),
                                  ["model", "label", "fairness_def", "var"]]

df_var_fairface_age = df_var_fairface.loc[(df_var_fairface["label"] == "age"),
                                  ["model", "fairness_def", "var"]]
df_var_fairface_gender = df_var_fairface.loc[(df_var_fairface["label"] == "gender"),
                                  ["model", "fairness_def", "var"]]
df_var_fairface_race = df_var_fairface.loc[(df_var_fairface["label"] == "race"),
                                  ["model", "fairness_def", "var"]]

##
df_var_utkface = df_var_trans.loc[(df_var_trans["dataset"] == "UTKFace"),
                                  ["model", "label", "fairness_def", "var"]]

df_var_utkface_age = df_var_utkface.loc[(df_var_utkface["label"] == "age"),
                                  ["model", "fairness_def", "var"]]
df_var_utkface_gender = df_var_utkface.loc[(df_var_utkface["label"] == "gender"),
                                  ["model", "fairness_def", "var"]]
df_var_utkface_race = df_var_utkface.loc[(df_var_utkface["label"] == "race"),
                                  ["model", "fairness_def", "var"]]

# full plot of all classifier on dataset
plot_var_fairface = (p9.ggplot(data=df_var_fairface, mapping = p9.aes(x="model", y="var", fill="label"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (8, 25))
        + p9.labs(x="ML model", y= "variance", size = 10, title=("Fairness evaluation on Fairface dataset"))
        # + p9.facet_grid(".~fairness_def")
        + p9.facet_wrap('fairness_def', ncol = 3)
        )
#print(plot_var_fairface)

# plot of each classifier on dataset
plot_var_fairface_race = (p9.ggplot(data=df_var_fairface_race, mapping = p9.aes(x="model", y="var"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (10, 20))
        + p9.labs(x="ML model", y= "variance", size = 10, title=("Race classification on Fairface dataset"))
        # + p9.facet_grid(".~fairness_def")
        + p9.facet_wrap('fairness_def', ncol = 3)
        )
#print(plot_var_fairface_race)
#plot_var_fairface_race.save(filename=r"C:\Users\tuanm\Desktop\MasterThesis\Output\Plots\final_plots\fairface_race.png")

plot_var_noft_avg = (p9.ggplot(data=df_var_trans_noft_avg, mapping = p9.aes(x="model", y="var_avg", fill="label"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (10, 10))
        + p9.labs(x="ML model", y= "variance", size = 10, title=("Average fairness evaluation"))
        # + p9.facet_grid(".~fairness_def")
        + p9.facet_wrap('fairness_def', ncol = 4)
        )
#print(plot_var_noft_avg)
#print(plot_var_facearg_race)

plot_dataset_composition = (p9.ggplot(data=df_dataset, mapping = p9.aes(x="dataset", y="percentage", fill="class"))
        + p9.geom_col(position="fill")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (10, 10))
        + p9.labs(x="", y= "percentage", size = 10, title=("Average fairness evaluation"))
        # + p9.facet_grid(".~fairness_def")
        + p9.facet_wrap('label', ncol = 4)
        )
#print(plot_dataset_composition)

df_dataset_age = df_dataset.loc[(df_dataset["label"] == "age"),
                                  ["dataset", "class", "percentage"]]
df_dataset_gender = df_dataset.loc[(df_dataset["label"] == "gender"),
                                  ["dataset", "class", "percentage"]]
df_dataset_race = df_dataset.loc[(df_dataset["label"] == "race"),
                                  ["dataset", "class", "percentage"]]

plot_dataset_composition = (p9.ggplot(data=df_dataset_race, mapping = p9.aes(x="dataset", y="percentage", fill="class"))
        + p9.geom_col(position="fill")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (6, 9))
        + p9.labs(x="", y= "ratio", size = 10, title=("Dataset composition race"))
        # + p9.facet_grid(".~fairness_def")
        #+ p9.facet_wrap('label', ncol = 4)
        )
print(plot_dataset_composition)