ECHO Training and predicting using KNN and SVM algorithms

python main.py data/English-train.xml data/English-dev.xml KNN-English.answer SVM-English.answer Best-English.answer English





ECHO Evaluating KNN results

scorer2 KNN-English.answer data/English-dev.key data/English.sensemap


ECHO Evaluating SVM results

scorer2 SVM-English.answer data/English-dev.key data/English.sensemap


ECHO Evaluating BEST SVM results (assignment 3b)

scorer2 Best-English.answer data/English-dev.key data/English.sensemap

