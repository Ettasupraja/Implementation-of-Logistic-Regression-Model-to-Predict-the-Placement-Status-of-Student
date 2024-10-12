# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

NAME: ETTA SUPRAJA
REG NO: 212223220022

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results. 
```
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
![Screenshot 2024-10-12 131500](https://github.com/user-attachments/assets/326896e6-9abe-4201-b803-083cf05be37c)


dataset.info()
![Screenshot 2024-10-12 131537](https://github.com/user-attachments/assets/69f1b4d5-c24c-402d-89a6-a84730f4fd9e)


dataset.drop('sl_no',axis=1)
dataset.info()
![Screenshot 2024-10-12 131604](https://github.com/user-attachments/assets/db9da1d2-8270-44d5-bbd2-0c9a8300b83b)

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
![Screenshot 2024-10-12 131639](https://github.com/user-attachments/assets/99bb3b54-a7a6-4d1e-a9b5-e6ab886f26fd)

dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset
![Screenshot 2024-10-12 131709](https://github.com/user-attachments/assets/fac55752-39af-4f3b-aa5c-b53aa20d6c89)


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
clf = LogisticRegression()
clf.fit(x_train,y_train)
![Screenshot 2024-10-12 131736](https://github.com/user-attachments/assets/e87442c2-e596-41ff-bb26-ce5bfba62d93)

X_train.shape
![Screenshot 2024-10-12 131811](https://github.com/user-attachments/assets/7bce60d8-2dbc-4519-9e29-133177e9cb49)

X_test.shape
![Screenshot 2024-10-12 131902](https://github.com/user-attachments/assets/551b9496-1a0c-4bc9-8a0b-e669052a4a25)

Y_train.shape
![Screenshot 2024-10-12 131931](https://github.com/user-attachments/assets/0bc2a42c-b192-469b-999a-983799dc9085)

Y_test.shape
![Screenshot 2024-10-12 131957](https://github.com/user-attachments/assets/b0c502bc-a9f0-47a5-8864-7a7a23444143)

y_pred=clf.predict(x_test)
y_pred
![Screenshot 2024-10-12 132028](https://github.com/user-attachments/assets/01ab4ae7-5be8-4d54-b077-bae63dd0853a)

from sklearn.metrics import confusion_matrix, accuracy_score
cf = confusion_matrix(y_test, y_pred)
print(cf)
![Screenshot 2024-10-12 132104](https://github.com/user-attachments/assets/3fd1d197-2f8f-45bb-a6bf-36366cb61cfb)


accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
![Screenshot 2024-10-12 132132](https://github.com/user-attachments/assets/73a84245-23f6-4d6b-a6fa-0d3d42cbeef0)

*/
```

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
