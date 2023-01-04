import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics import mean_squared_error
import numpy as np
import data
import matplotlib.pyplot as plt

x,y = data.pop_nans([data.listing_description],data.scores_rating)
y=y.to_numpy()
text = x[0]

#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

#TF-IDF
vectorizer = TfidfVectorizer(stop_words = "english", max_df=0.2)
X = vectorizer.fit_transform(text)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
mean_error=[]; std_error=[]
neighbor_range  = [5,10,15,20,25]
for n in neighbor_range:
    model = KNeighborsRegressor(n_neighbors=n,weights="distance").fit(X,y)
    temp=[]
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp.append(mean_squared_error(y[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

#print(mean_error)
#print(std_error)
#plot
plt.errorbar(neighbor_range,mean_error,yerr=std_error)
plt.xlabel('NN'); plt.ylabel("Mean square error")
plt.title("KNN Hyperparameter Cross-Validation")
plt.show()


"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)

model = KNeighborsRegressor(n_neighbors=2,weights='distance').fit(X_train,y_train)
ypred = model.predict(X_test)
print(mean_squared_error(y_test,ypred))
"""