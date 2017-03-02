ECHO Running Assignment 3B

#python main.py data/English-train.xml data/English-dev.xml KNN-English.answer SVM-English.answer Best-English.answer English

ECHO Scoring ...
scorer2 Best-English.answer data/English-dev.key data/English.sensemap
 
