import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import ratings_plot
from sklearn.dummy import DummyRegressor

#Read datasets
df = pd.read_csv('reviews_num.csv', sep=',',skiprows=1)
reviews_num = df.iloc[:,0]
df = pd.read_csv('superhost.csv', sep=',',skiprows=1)
is_superhost = df.iloc[:,0]
df = pd.read_csv('value.csv', sep=',',skiprows=1)
value_score = df.iloc[:,0]
df = pd.read_csv('neighbourhoods.csv', sep=',',skiprows=1)
neighbourhoods_sentiment = df.iloc[:,0]
df = pd.read_csv('rating.csv', sep=',',skiprows=1)
y = df.iloc[:,0]

"""ratings_plot.plot_2d(value_score,y,'Overall Rating', 'Value Score','Value Sentiment vs Rating')
ratings_plot.plot_2d(is_superhost,y,'Overall Rating','Is Superhost','Superhost status vs Rating')
ratings_plot.plot_2d(neighbourhoods_sentiment,y,'Overall Rating','Neighbourhood Sentiment','Neighbourhood sentiments vs Rating')"""

X=np.column_stack((is_superhost,value_score,neighbourhoods_sentiment))

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
plt.rc('font',size=18); plt.rcParams['figure.constrained_layout.use'] = True
mean_error = []; std_error = []
q_range = [1,2,3,4]
for q in q_range:
    from sklearn.preprocessing import PolynomialFeatures
    Xpoly = PolynomialFeatures(q).fit_transform(X)
    model = LinearRegression()
    temp = []; plotted = False
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train],y[train])
        ypred = model.predict(Xpoly[test])
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(y[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
#print(mean_error)
#print(std_error)
plt.errorbar(q_range,mean_error,yerr=std_error,linewidth=3)
plt.xlabel('q')
plt.ylabel('Mean square error')
plt.title('Cross-Validation of Polynomial Features')
plt.show()

model = LinearRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
model.fit(X_train,y_train)
ypred = model.predict(X_test)
print(mean_squared_error(y_test,ypred))

"""from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5,weights='uniform').fit(X_train,y_train)
ypred = model.predict(X_test)
print(mean_squared_error(y_test,ypred))"""

from sklearn.dummy import DummyRegressor
dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
ypred = dummy.predict(X_test)
print(mean_squared_error(y_test,ypred))




