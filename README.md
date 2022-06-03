# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import Logistic regression from sklearn.
4. Assign the points for representing in the graph
5. Predict the regression for salary by using the representation of the graph.
6. Compare the graphs and hence we obtained the logistic regression for the given datas.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by : Iniyan S
RegisterNumber : 212220040053
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets = pd.read_csv('/content/sample_data/Social_Network_Ads (1).csv')
X=datasets.iloc[:,[2,3]].values
Y=datasets.iloc[:,4].values 

from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test= train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_X

X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_Train,Y_Train)

Y_Pred =classifier.predict(X_Test)
Y_Pred

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(Y_Test,Y_Pred)
cm

from sklearn import metrics
accuracy= metrics.accuracy_score(Y_Test,Y_Pred)
accuracy

recall_sensitivity = metrics.recall_score(Y_Test,Y_Pred,pos_label =1)
recall_specificity = metrics.recall_score(Y_Test,Y_Pred,pos_label =0)
recall_sensitivity,recall_specificity

from matplotlib.colors import ListedColormap
X_set,Y_set=X_Train,Y_Train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('black','yellow')))

plt.xlim(X1.min(),X2.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
  plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],c=ListedColormap(('white','yellow'))(i),label=j)
  plt.title('Logistic Regression(Training set)')
  plt.xlabel('Age')
  plt.ylabel('Estimated Salary')
  plt.legend()
  plt.show()

```

## Output:
Y_Pred:
![OP1](/OP1.png)

Confusion_matrix:
![OP2](/OP2.png)

Accuracy:
![OP3](/OP3.png)

Recall_sensitivity,Recall_specificity:
![OP4](/OP4.png)

Logistic Regression:
![OP5](/OP5.png)

![OP6](/OP6.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

