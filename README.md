# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: A.J.PRANAV
RegisterNumber: 212222230107  
*/
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

df=pd.read_csv('/content/drive/My Drive/student_scores.csv')
df

df.head()

df.tail()

X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
'''
print(df)
print(X_test)
print(Y_test)
'''
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Training set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
plt.scatter(X_test,Y_test,color='blue')
plt.plot(X_test,reg.predict(X_test),color='silver')
plt.title("Test set(H vs S)")
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

a=np.array([[10]])
Y_pred1=reg.predict(a)
print(Y_pred1)
```

## Output:
### df

![image](https://github.com/Pranav-AJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118904526/305e2d11-a0d3-490e-96d0-6af34fac033d)

### df.head()

![image](https://github.com/Pranav-AJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118904526/05f6ebe1-5a1c-4c61-a37b-c7af5cbf2306)

### df.tail()

![image](https://github.com/Pranav-AJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118904526/fd2ab00c-4655-4a33-af6b-de7ea7299772)

![image](https://github.com/Pranav-AJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118904526/2414450a-6635-4942-a929-e6122b504519)

![image](https://github.com/Pranav-AJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118904526/100a1852-71ce-4954-b6c5-341711537732)

![image](https://github.com/Pranav-AJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118904526/1fca3c7e-1481-43c0-99cc-8948af9b7a83)

![image](https://github.com/Pranav-AJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118904526/5d163a49-52c3-4733-a29d-d47bba73be78)

![image](https://github.com/Pranav-AJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118904526/e8150677-8181-4d38-a1da-9ed9964e368d)

![image](https://github.com/Pranav-AJ/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118904526/49eb51c8-e1d8-4c4c-827c-300c3a2646ac)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
