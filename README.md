# Implementation-of-Linear-Regression-Using-Gradient-Descent->

## AIM :
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required :
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :
1.Start the program.

2.Import numpy as np.

3.Give the header to the data.

4.Find the profit of population.

5.Plot the required graph for both for Gradient Descent Graph and Prediction Graph.

6.End the program.

## Program :
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Pradeep Raj P
RegisterNumber:  212222240073
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt" , header=None )

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step =5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10.000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")


def computeCost(X,y,theta):
  """
  Take in a numpy array X,Y,THETA and generate the cost function using the    in a linear regression model

  """
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2

  return 1/(2*m)*np.sum(square_err)


data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iterations")
plt.ylabel("$J(\Theta)$")
plt.title("Cost Function Of Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="green")
plt.xticks(np.arange(5,30,step =5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10.000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70,000, we predict a profit of $"+str(round(predict2,0)))


```


## Output:

### PROFIT PREDICTION GRAPH:

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707347/ab51c031-e56a-4f7c-ac1c-d1d4138c9131)

### COMPUTE COST VALUE:

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707347/b7ed77b6-4b6e-4d81-b1c9-a64581967244)

### h(x) VALUE:

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707347/ee900b69-d3e5-4aab-bc77-8c622eff4ce9)

### COST FUNCTION USING GRADIENT DESCENT GRAPH:

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707347/2e05ee2c-c460-4665-a845-2dbad8eca97d)


### PROFIT PREDICTION GRAPH:

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707347/e6c39a22-c750-4ef5-b861-cf56050cd2c5)

### PROFIT FOT THE POPULATION 35,000:

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707347/f0faf99d-a3cc-4275-b123-481bbc0bcae5)

### PROFIT FOR THE POPULATION 70,000:

![image](https://github.com/Pradeeppachiyappan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707347/87c341c5-ae1d-44f4-b8c7-65bbdc255a3c)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
