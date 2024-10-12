# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

NAME: ETTA SUPRAJA
REG NO: 212223220022

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ETTA SUPRAJA
RegisterNumber: 212223220022
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataset = pd.read_csv('Placement_Data_Full_Class (1).csv')
dataset

![image](https://github.com/user-attachments/assets/8aa21a6e-1810-4acb-83da-d995b610d0fa)

dataset.info()
![image](https://github.com/user-attachments/assets/5de7b7a2-6657-4bb2-a8b3-5b0506f94e12)

dataset.drop('sl_no',axis=1)
dataset.info()
![image](https://github.com/user-attachments/assets/ef7e7d89-6702-4c4f-96fc-9c1d663c3182)

dataset["gender"]= dataset["gender"].astype('category')
dataset["ssc_b"]= dataset["ssc_b"].astype('category')
dataset["hsc_b"]= dataset["hsc_b"].astype('category')
dataset["hsc_s"]= dataset["hsc_s"].astype('category')
dataset["degree_t"]= dataset["degree_t"].astype('category')
dataset["workex"]= dataset["workex"].astype('category')
dataset["specialisation"]= dataset["specialisation"].astype('category')
dataset["status"]= dataset["status"].astype('category')
dataset.dtypes
dataset.info()
![image](https://github.com/user-attachments/assets/59769299-b973-4e74-b8a6-7075610d4239)

dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset
![image](https://github.com/user-attachments/assets/8334422d-e224-495f-ac72-960f8e384b2a)


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
clf = LogisticRegression()
clf.fit(x_train,y_train)
![image](https://github.com/user-attachments/assets/60a23258-0523-4338-8cf0-c1fa83a43b2c)

X_train.shape
![image](https://github.com/user-attachments/assets/f2913b2b-228d-422a-992b-aa92a4724a53)

X_test.shape
![image](https://github.com/user-attachments/assets/1e849e14-1da0-4c41-8a1e-cf5e30d88b1b)

Y_train.shape
![image](https://github.com/user-attachments/assets/c8bc9b46-3265-4ccf-ba22-3fcec9021bf8)

Y_test.shape
![image](https://github.com/user-attachments/assets/15a71490-f209-4cd6-a377-6214bc538e57)

y_pred=clf.predict(x_test)
y_pred
![image](https://github.com/user-attachments/assets/cf7b43e7-34d7-4ff3-8ae8-7a17daafa326)

from sklearn.metrics import confusion_matrix, accuracy_score
cf = confusion_matrix(y_test, y_pred)
print(cf)
![image](https://github.com/user-attachments/assets/080831e6-ba42-4538-9ba2-fb8a4384b45e)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
![image](https://github.com/user-attachments/assets/85270d85-4e7d-4428-8023-f5f2c223dd56)

*/
```


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
