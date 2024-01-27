import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle



df=pd.read_excel('/Users/roshdyhamdy/Desktop/FinalProject/pesplanus_dataset1.xlsx')
train,valid,test=np.split(df,[int(0.6*len(df)),int(0.8*len(df))])
#print(df.head())

def scale_dataset(DataFrame,oversample=False):

    x=DataFrame[DataFrame.columns[:-1]].values
    y=DataFrame[DataFrame.columns[-1]].values
    
    data=np.hstack((x, np.reshape(y,(-1,1))))
    return data,x,y

train,x_train,y_train=scale_dataset(train,oversample=True)
valid,x_valid,y_valid=scale_dataset(valid,oversample=False)
test,x_test,y_test=scale_dataset(test,oversample=False)



svm_model=SVC()
svm_model.fit(x_train,y_train)

pickle.dump(svm_model, open("/Users/roshdyhamdy/Desktop/FinalProject/model.pkl", "wb"))



