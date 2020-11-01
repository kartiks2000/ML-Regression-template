# REGRESSION TEMPLATE

# Importing liberaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Importing dataset
dataset = pd.read_csv("Position_Salaries.csv")



# Seperating dependent and independent variables
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values



 
# Splitting the dataset into Training set and Test set
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
'''



# Feature Scaling -> Normalizing the range of data/vairiable values
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
'''




# Fitting regression model to the dataset
# Create your regressor here......





# Predicting a new result with Regression
y_pred = predict([[6.5]])




# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, predict(x), color = 'blue')
plt.title('Truth or Bluff (Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# Visualising the Regression results (for higher resolution and smoother curve)
# If we want the graph to be more accurate by making inputs complicated.
# It gives a much smoother curve.
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
