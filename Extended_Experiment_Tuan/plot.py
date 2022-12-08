# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 07:21:06 2022

@author: tuanm
"""

import plotnine as p9
import pandas as pd

df_accuracy_score_extended = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\Excel_New\final\accuracy_score_extended.csv")
df_avg_accuracy_score_extended = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\Excel_New\final\avg_accuracy_score_extended.csv")
df_var_all_extended = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\Excel_New\final\var_all_trans_extended.csv")
df_var_sum_extended = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\Excel_New\final\var_sum_extended.csv")

df_acc_score_ex2 = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\Excel_New\final\Experiment2_acc_score.csv")
df_var_ex2 = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\Excel_New\final\Experiment2_var_allutk.csv")
df_var_sum_ex2 = pd.read_csv(r"C:\Users\tuanm\Desktop\MasterThesis\Output\Excel_New\final\Experiment2_var_sum.csv")

df_acc_score_race = df_accuracy_score_extended.loc[(df_accuracy_score_extended["label"] == "race"),
                                  ["model", "dataset", "accuracy_score"]]
df_var_ex2_race = df_var_ex2.loc[(df_var_ex2["label"] == "race"),["model", "dataset", "fairness_def", "var"]]
df_var_fairface = df_var_all_extended.loc[(df_var_all_extended["dataset"] == "Fairface"),
                                          ["model", "label", "fairness_def", "var"]]
df_var_fairface_race = df_var_fairface.loc[(df_var_all_extended["label"] == "race"),
                                          ["model", "fairness_def", "var"]]
df_var_race = df_var_all_extended.loc[(df_var_all_extended["label"] == "race"),
                                          ["model", "dataset", "fairness_def", "var"]]

plot_accuracy = (p9.ggplot(data=df_accuracy_score_extended, mapping = p9.aes(x="model", y="accuracy_score", fill="dataset"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (10, 4))
        + p9.labs(x="ML model", y= "accuracy score", title=("Accuracy for each classification on all datasets"))
        + p9.facet_grid(".~label")
        )

plot_avg_accuracy = (p9.ggplot(data=df_avg_accuracy_score_extended, mapping = p9.aes(x="model", y="avg acc"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (10, 4))
        + p9.labs(x="ML model", y= "accuracy score", title=("Average accuracy for each classification"))
        + p9.facet_grid(".~label")
        )

plot_acc_score_race = (p9.ggplot(data=df_acc_score_race, mapping = p9.aes(x="model", y="accuracy_score", fill="dataset"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (12, 12))
        + p9.labs(x="ML model", y= "accuracy score", size = 10, title=("Experiment 3: Accuracy score of race classification"))
        # + p9.facet_grid(".~fairness_def")
        # + p9.facet_wrap('fairness_def', ncol = 3)
        )

plot_acc_score_race_ex2 = (p9.ggplot(data=df_acc_score_ex2, mapping = p9.aes(x="model", y="accuracy_score", fill="dataset"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (12, 12))
        + p9.labs(x="ML model", y= "accuracy score", size = 10, title=("Experiment 2: Accuracy score of race classification"))
        # + p9.facet_grid(".~fairness_def")
        # + p9.facet_wrap('fairness_def', ncol = 3)
        )

plot_var_ex2 = (p9.ggplot(data=df_var_ex2_race, mapping = p9.aes(x="model", y="var", fill="dataset"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (10, 10))
        + p9.labs(x="ML model", y= "variance", size = 10, title=("Experiment 2: Fairness evaluation on classification race"))
        # + p9.facet_grid(".~fairness_def")
        + p9.facet_wrap('fairness_def', ncol = 3)
        )

plot_var_sum = (p9.ggplot(data=df_var_sum_ex2, mapping = p9.aes(x="model", y="sum of var", fill="dataset"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (10, 8))
        + p9.labs(x="ML model", y= "accuracy score", title=("Experiment 2: Var sum for each classification on all datasets"))
        + p9.facet_grid(".~label")
        )

plot_var_fairface = (p9.ggplot(data=df_var_race, mapping = p9.aes(x="model", y="var", fill="dataset"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (10, 15))
        + p9.labs(x="ML model", y= "variance", size = 10, title=("Experiment 3:Fairness evaluation on classification race of Fairface dataset"))
        # + p9.facet_grid(".~fairness_def")
        + p9.facet_wrap('fairness_def', ncol = 3)
        )

plot_var_sum = (p9.ggplot(data=df_var_sum_extended, mapping = p9.aes(x="model", y="sum of var", fill="dataset"))
        + p9.geom_col(position="dodge")
        + p9.theme(axis_text_x = p9.element_text(angle=90), figure_size = (10, 4))
        + p9.labs(x="ML model", y= "accuracy score", title=("Experiment 3: Var sum for each classification on all datasets"))
        + p9.facet_grid(".~label")
        )

print(plot_var_fairface)