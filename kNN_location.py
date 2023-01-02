import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import data

#kNN Regression model for location scores
x1 = data.listings_longitude
x2 = data.listings_latitude
y = data.scores_location
for i in range(len(data.scores_location)):
    if math.isnan(data.scores_location[i]):
        x1.pop(i)
        x2.pop(i)
        y.pop(i)

y = y.to_numpy()
X=np.column_stack((x1,x2))


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],y)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=10/len(X),shuffle=True)
model = KNeighborsRegressor(n_neighbors=10,weights='distance').fit(X_train,y_train)
#Prediction
ypred = model.predict(X_test)
print(ypred)
print(y_test)



