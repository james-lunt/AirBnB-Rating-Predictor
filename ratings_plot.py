import data
import matplotlib.pyplot as plt

#Scatter plot of listing ratings
plt.rc('font', size=10)
plt.xlabel('Listings numbered'); plt.ylabel('Score rating')
plt.title('Scatter plot of listing ratings')
plt.rcParams['figure.constrained_layout.use'] = True
plt.scatter(list(range(0, data.num_listings)),data.scores_rating, color='green', s=2)
plt.show()
