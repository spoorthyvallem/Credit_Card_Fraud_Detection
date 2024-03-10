import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
ccdata=pd.read_csv("C:/Users/Spoorthy/Desktop/Grad/Projects/CreditCardFraudDetection/creditcard.csv")
le=ccdata[ccdata.Class==0]
fr=ccdata[ccdata.Class==1]
le_sample=le.sample(n=492)

#concatinate le_sample and fr
new_ccd=pd.concat([le_sample,fr],axis=0)
X=new_ccd.drop(columns='Class',axis=1)
Y=new_ccd['Class']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
model=LogisticRegression()
#training the Logistic Regression model with training data
model.fit(x_train,y_train)
x_train_pred=model.predict(x_train)
traind_acc=accuracy_score(x_train_pred,y_train)
x_test_pred=model.predict(x_test)
testd_acc=accuracy_score(x_test_pred,y_test)
st.title("Credit Card Fraud Detection")
input=st.text_input("Enter All Required Features Values")
input_splited=input.split(',')
buton=st.button("Submit")
if buton:
    feature=np.asarray(input_splited,dtype=np.float64)
    pred=model.predict(feature.reshape(1,-1))
    if pred[0]==0:
        st.write("It is a Legitimate Transaction")
    else:
        st.write("It is a Fraudulent Transaction")