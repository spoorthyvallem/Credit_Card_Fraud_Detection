import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

#loading dataset to pandas dataets
ccdata=pd.read_csv("C:/Users/Spoorthy/Desktop/Grad/Projects/CreditCardFraudDetection/creditcard.csv")

ccdata.head()

ccdata.tail()

ccdata.info()

ccdata.isnull().sum()

ccdata.shape

"""**Central** **Tendancies**
1.Mean
2.Median
3.Mode
"""

#handling missing data by analysing the distribution
# fig,a=plt.subplots(fidsize=(8,8))
# sns.distplot(ccdata.Amount)

#distribution of legit transaction &fradulent transaction
ccdata['Class'].value_counts()

"""**This dataset is highly** **unbalanced**
0-> normal
1-> fraud
"""

#separate data for analysis
le=ccdata[ccdata.Class==0]
fr=ccdata[ccdata.Class==1]
print(le.shape)
print(fr.shape)

#statistical measures of data
le.Amount.describe()

fr.Amount.describe()

#compare values for both transactions
ccdata.groupby('Class').mean()

"""**Undersampling**
Build a sample dataset containing similar distribution of normal and fraud transactions
"""

le_sample=le.sample(n=492)

#concatinate le_sample and fr
new_ccd=pd.concat([le_sample,fr],axis=0)

new_ccd.head()

new_ccd.tail()

new_ccd['Class'].value_counts()

new_ccd.groupby('Class').mean()

"""Splitting the data into features & targets"""

X=new_ccd.drop(columns='Class',axis=1)
Y=new_ccd['Class']
print(X)

print(Y)

"""Splitting datA into train and test"""

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,x_train.shape,x_test.shape)

"""**Model** **Training**
**logistic regression**
"""

model=LogisticRegression()
#training the Logistic Regression model with training data
model.fit(x_train,y_train)

"""Model Evaluation

Accuracy Score
"""

x_train_pred=model.predict(x_train)
traind_acc=accuracy_score(x_train_pred,y_train)
print('Accuracy on Training data: ',traind_acc)

x_test_pred=model.predict(x_test)
testd_acc=accuracy_score(x_test_pred,y_test)
print('Accuracy on Testing data: ',testd_acc)

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