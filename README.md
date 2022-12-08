# Masterthesis

Repository for the code of our Masterthesis "The influence of deep neural network architectures on the classification fairness of face recognition tasks". 

## Models used

| Name | Owner |Link |
| --- | --- | --- |
| Deepface | Serengil | https://github.com/serengil/deepface |
| Fairface | Kärkkäinen, Joo | https://github.com/dchen236/FairFace |
| Face_Recognition | Bansal | https://github.com/ChayanBansal/Face_Recognition |
| AGR-recognition | Kumar | https://github.com/kuabhish/Age-Gender-Race-recognition |


## Datasets used

| Name |  Size | Attributes | Link |
| --- | --- | --- | --- |
| UTKFace | 20k+ | age, gender, ethnicity | https://susanqq.github.io/UTKFace/ |
| Fairface | 100k+ | age groups, gender, ethnicity | https://www.kaggle.com/datasets/lantian773030/fairface |
| Face_Recognition.json | 100+ | age groups, gender, ethnicity, emotion | https://github.com/ChayanBansal/Face_Recognition/tree/master/dataset |
| FaceARG | 100+ | age groups, gender, ethnicity, emotion | https://www.cs.ubbcluj.ro/~dadi/FaceARG-database.html |

## Additional experiment
## Tuan:
Can be found in folder Extended_Experiment_Tuan where each folder consists one model.
- Fairness.py used for fairness evaluation
- plot.py used to plot results
Each folder has one file for prediction (i.e. AGR_ft.py), one file with the architecture of the model (my_model.py) and one file to train the model (train_ft.py).
Additionally you can find weights trained on the dataset Fairface or UTKFace as .h5 files (i.e. utk_newage_conv1.h5 which means architecture AGR_conv1 trained on UTKFace).
- Results can be found in Extended_Experiment_Tuan/Output folder. 
