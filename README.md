# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="darkseagreen")
plt.plot(x_train,regressor.predict(x_train),color="plum")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

y_pred


#graph plot for test data
plt.scatter(x_test,y_test,color="darkblue")
plt.plot(x_test,regressor.predict(x_test),color="plum")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![Screenshot (25)](https://github.com/MaheshMuthuL/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135570619/6917a59e-d5b2-4cb8-ad9d-c89935d0db36)
![Screenshot (26) - Copy](https://github.com/MaheshMuthuL/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135570619/684a65f5-5b24-4cde-9a97-5bb12621d8c8)
![Screenshot (27)](https://github.com/MaheshMuthuL/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135570619/a8cda932-d4e8-4ad6-97bf-09e36c0abebb)
![Screenshot (28)](https://github.com/MaheshMuthuL/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135570619/2d2d83d8-efd7-4fa4-a835-e7ea5eb85728)
![Screenshot (32)](https://github.com/MaheshMuthuL/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135570619/f0ce2721-6e60-4df7-9a8f-3b1380581db9)
![Screenshot (35)](https://github.com/MaheshMuthuL/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135570619/819d7d3a-6d5e-4430-84b2-b9962d477946)
![Screenshot (30)](https://github.com/MaheshMuthuL/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135570619/fa83ea33-405f-449e-8680-62f611cfee5f)
![Screenshot (34)](https://github.com/MaheshMuthuL/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135570619/1e72f794-0a30-43a5-89b9-311570e46bfb)
![Screenshot (31)](https://github.com/MaheshMuthuL/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135570619/877d66fe-ef68-43b6-a580-61880437a46d)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
