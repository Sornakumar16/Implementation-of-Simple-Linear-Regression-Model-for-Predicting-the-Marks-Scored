# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: sornakumar s
RegisterNumber: 212223230210 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores (1).csv")
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
import matplotlib.pyplot as plt
plt.scatter(X_train,Y_train,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)










```

## Output:
### df.head()
![image](https://github.com/Sornakumar16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849327/563a2025-2ad4-4aa8-a0fd-374a5fb8fb77)
### df.tail()
![image](https://github.com/Sornakumar16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849327/47f63e2c-8319-4cbf-9483-d58280dc6991)
### x values
![image](https://github.com/Sornakumar16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849327/39cfeac7-a84f-4265-8797-869e911bbeeb)
### y values
![image](https://github.com/Sornakumar16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849327/762c40a8-d37b-4df9-bc29-203a234486a0)
### y predicted values


![image](https://github.com/Sornakumar16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849327/da4c68db-c2be-4280-bdc1-8f7e71478dcd)
### y test values
![image](https://github.com/Sornakumar16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849327/f714b4b8-debe-4747-b6cd-741ff3c1e1d3)
### training set graph
![image](https://github.com/Sornakumar16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849327/67b9345e-f324-45cc-8175-1ea3a6b2bb46)

### testing set graph
![image](https://github.com/Sornakumar16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849327/55db52e0-6cd6-4a43-9759-3f47f3bd9f96)
### values of MSE,MAE,RMSE
![image](https://github.com/Sornakumar16/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849327/93440181-f3ca-454f-bf4c-97a1db3de450)









## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
