import data
import matplotlib.pyplot as plt
import math
import numpy as np
import csv
"""#Scatter plot of listing ratings
plt.rc('font', size=10)
plt.xlabel('Listings numbered'); plt.ylabel('Score rating')
plt.title('Scatter plot of listing ratings')
plt.rcParams['figure.constrained_layout.use'] = True
plt.scatter(list(range(0, data.num_listings)),data.scores_accuracy, color='green', s=2)
plt.show()
"""


y = []
for id in data.reviews_listing_id:
    for i in range(len(data.listings_id)):
        if id == data.listings_id[i]:
            y.append(data.scores_accuracy[i])

with open('filepath.csv', 'w') as f:
    writer = csv.writer(f)
    for ys in y:
        csv.writerow(ys)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english', max_df=0.2)
X = vectorizer.fit_transform(data.reviews_comments.values.astype(str))

from scipy import spatial
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/len(X),shuffle=True)
result = 1 - spatial.distance.cosine(X_train, X_test)

#Scatter plot of listing ratings
plt.rc('font', size=10)
plt.xlabel('Similarity s(v,w)'); plt.ylabel('Accuracy rating')
plt.title('Similarity score against accuracy rating')
plt.rcParams['figure.constrained_layout.use'] = True
plt.scatter(result,data.scores_accuracy, color='green', s=2)
plt.show()
