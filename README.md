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
sl_no	gender	ssc_p	ssc_b	hsc_p	hsc_b	hsc_s	CGPA	degree_t	workex	etest_p	specialisation	Masters	status
0	1	M	67.00	Others	91.00	Others	Commerce	58.00	Sci&Tech	No	55.00	Mkt&HR	58.80	Placed
1	2	M	79.33	Central	78.33	Others	Science	77.48	Sci&Tech	Yes	86.50	Mkt&Fin	66.28	Placed
2	3	M	65.00	Central	68.00	Central	Arts	64.00	Comm&Mgmt	No	75.00	Mkt&Fin	57.80	Placed
3	5	M	85.80	Central	73.60	Central	Commerce	73.30	Comm&Mgmt	No	96.80	Mkt&Fin	55.50	Placed
4	8	M	82.00	Central	64.00	Central	Science	66.00	Sci&Tech	Yes	67.00	Mkt&Fin	62.14	Placed
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
210	199	F	67.00	Central	70.00	Central	Commerce	65.00	Others	No	88.00	Mkt&HR	71.96	Not Placed
211	202	M	54.20	Central	63.00	Others	Science	58.00	Comm&Mgmt	No	79.00	Mkt&HR	58.44	Not Placed
212	207	M	41.00	Central	42.00	Central	Science	60.00	Comm&Mgmt	No	97.00	Mkt&Fin	53.39	Not Placed
213	209	F	43.00	Central	60.00	Others	Science	65.00	Comm&Mgmt	No	92.66	Mkt&HR	62.92	Not Placed
214	215	M	62.00	Central	58.00	Others	Science	53.00	Comm&Mgmt	No	89.00	Mkt&HR	60.22	Not Placed
215 rows × 14 columns


dataset.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 215 entries, 0 to 214
Data columns (total 14 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   sl_no           215 non-null    int64  
 1   gender          215 non-null    object 
 2   ssc_p           215 non-null    float64
 3   ssc_b           215 non-null    object 
 4   hsc_p           215 non-null    float64
 5   hsc_b           215 non-null    object 
 6   hsc_s           215 non-null    object 
 7   CGPA            215 non-null    float64
 8   degree_t        215 non-null    object 
 9   workex          215 non-null    object 
 10  etest_p         215 non-null    float64
 11  specialisation  215 non-null    object 
 12  Masters         215 non-null    float64
 13  status          215 non-null    object 
dtypes: float64(5), int64(1), object(8)
memory usage: 23.6+ KB


dataset.drop('sl_no',axis=1)
dataset.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 215 entries, 0 to 214
Data columns (total 14 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   sl_no           215 non-null    int64  
 1   gender          215 non-null    object 
 2   ssc_p           215 non-null    float64
 3   ssc_b           215 non-null    object 
 4   hsc_p           215 non-null    float64
 5   hsc_b           215 non-null    object 
 6   hsc_s           215 non-null    object 
 7   CGPA            215 non-null    float64
 8   degree_t        215 non-null    object 
 9   workex          215 non-null    object 
 10  etest_p         215 non-null    float64
 11  specialisation  215 non-null    object 
 12  Masters         215 non-null    float64
 13  status          215 non-null    object 
dtypes: float64(5), int64(1), object(8)
memory usage: 23.6+ KB

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
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 215 entries, 0 to 214
Data columns (total 14 columns):
 #   Column          Non-Null Count  Dtype   
---  ------          --------------  -----   
 0   sl_no           215 non-null    int64   
 1   gender          215 non-null    category
 2   ssc_p           215 non-null    float64 
 3   ssc_b           215 non-null    category
 4   hsc_p           215 non-null    float64 
 5   hsc_b           215 non-null    category
 6   hsc_s           215 non-null    category
 7   CGPA            215 non-null    float64 
 8   degree_t        215 non-null    category
 9   workex          215 non-null    category
 10  etest_p         215 non-null    float64 
 11  specialisation  215 non-null    category
 12  Masters         215 non-null    float64 
 13  status          215 non-null    category
dtypes: category(8), float64(5), int64(1)
memory usage: 12.9 KB

dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset
	sl_no	gender	ssc_p	ssc_b	hsc_p	hsc_b	hsc_s	CGPA	degree_t	workex	etest_p	specialisation	Masters	status
0	1	1	67.00	1	91.00	1	1	58.00	2	0	55.00	1	58.80	1
1	2	1	79.33	0	78.33	1	2	77.48	2	1	86.50	0	66.28	1
2	3	1	65.00	0	68.00	0	0	64.00	0	0	75.00	0	57.80	1
3	5	1	85.80	0	73.60	0	1	73.30	0	0	96.80	0	55.50	1
4	8	1	82.00	0	64.00	0	2	66.00	2	1	67.00	0	62.14	1
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
210	199	0	67.00	0	70.00	0	1	65.00	1	0	88.00	1	71.96	0
211	202	1	54.20	0	63.00	1	2	58.00	0	0	79.00	1	58.44	0
212	207	1	41.00	0	42.00	0	2	60.00	0	0	97.00	0	53.39	0
213	209	0	43.00	0	60.00	1	2	65.00	0	0	92.66	1	62.92	0
214	215	1	62.00	0	58.00	1	2	53.00	0	0	89.00	1	60.22	0
215 rows × 14 columns

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
clf = LogisticRegression()
clf.fit(x_train,y_train)
LogisticRegression
LogisticRegression()

X_train.shape
(172, 13)

X_test.shape
(43, 13)

Y_train.shape
(172,)


Y_test.shape

(43,)

y_pred=clf.predict(x_test)
y_pred

array([0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1,
       0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
      dtype=int8)


from sklearn.metrics import confusion_matrix, accuracy_score
cf = confusion_matrix(y_test, y_pred)
print(cf)

[[12  3]
 [ 1 27]]

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

0.9069767441860465

*/
```

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
