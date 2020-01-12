#!/usr/bin/env python3

#Diabetes Prediction Using Support Vector Classifier
import pickle
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

#For training
def train():
    dataset = pd.read_csv('diabetes.csv')
    X = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    Y = dataset[['Outcome']]
    
    #train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 101)
    
    from sklearn.svm import SVC
    model = SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
  
    svc=model.fit(X_train,Y_train.values.ravel())
    
    #Save Model As Pickle File
    with open('svc.pkl','wb') as m:
        pickle.dump(svc,m)
    test(X_test,Y_test)

#Test accuracy of the model
def test(X_test,Y_test):
    with open('svc.pkl','rb') as mod:
        p=pickle.load(mod)
    

def find_data_file(filename):
    if getattr(sys, "frozen", False):
        # The application is frozen.
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen.
        datadir = os.path.dirname(__file__)

    return os.path.join(datadir, filename)


def check_input(data) ->int :
    df=pd.DataFrame(data=data,index=[0])
    with open(find_data_file('svc.pkl'),'rb') as model:
        p=pickle.load(model)
    op=p.predict(df)
    return op[0]
if __name__=='__main__':
    train()    
