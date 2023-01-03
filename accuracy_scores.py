import data
import matplotlib.pyplot as plt
import math
import numpy as np
import csv

id_to_accuracy = {id: score for id, score in zip(data.listings_id, data.scores_accuracy)}
y = [id_to_accuracy[id] for id in data.reviews_listing_id]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english', max_df=0.2)
X = vectorizer.fit_transform(data.reviews_comments.values.astype(str))

from scipy import spatial
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.0001,shuffle=True)
print(X_train)
print(X_test)
result = 1 - spatial.distance.cosine(X_train, X_test)

#Scatter plot of listing ratings
plt.rc('font', size=10)
plt.xlabel('Similarity s(v,w)'); plt.ylabel('Accuracy rating')
plt.title('Similarity score against accuracy rating')
plt.rcParams['figure.constrained_layout.use'] = True
plt.scatter(result,data.scores_accuracy, color='green', s=2)
plt.show()
