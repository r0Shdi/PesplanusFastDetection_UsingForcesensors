from typing import Tuple,List
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pickle



df=pd.read_excel('/Users/roshdyhamdy/Desktop/FinalProject/pesplanus_dataset1.xlsx')
train,valid,test=np.split(df,[int(0.6*len(df)),int(0.8*len(df))])
#print(df.head())

def scale_dataset(DataFrame,oversample=False):

    x=DataFrame[DataFrame.columns[:-1]].values
    y=DataFrame[DataFrame.columns[-1]].values
    scaler=StandardScaler()
    x=scaler.fit_transform(x)
    if oversample:
      ros=RandomOverSampler()
      x,y=ros.fit_resample(x,y)
    data=np.hstack((x, np.reshape(y,(-1,1))))
    return data,x,y

train,x_train,y_train=scale_dataset(train,oversample=True)
valid,x_valid,y_valid=scale_dataset(valid,oversample=False)
test,x_test,y_test=scale_dataset(test,oversample=False)


classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train,y_train)


pickle.dump(classifier, open("model.pkl", "wb"))


print(classifier.predict(x_test))
print(x_test)
