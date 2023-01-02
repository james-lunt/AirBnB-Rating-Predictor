import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import data

#kNN Regression model for location scores
x1 = data.listings_longitude
x2 = data.listings_latitude
y = data.scores_location

inputs,outputs = data.pop_nans((x1,x2),y)
y = outputs.to_numpy()
X=np.column_stack((inputs[0],inputs[1]))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],y)
ax.set_title('Location Rating vs Listing Coordinates')
ax.set_xlabel('Logitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Location Rating')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=10/len(X),shuffle=True)
model = KNeighborsRegressor(n_neighbors=10,weights='distance').fit(X_train,y_train)
#Prediction
ypred = model.predict(X_test)
print(ypred)
print(y_test)



