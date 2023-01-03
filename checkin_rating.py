import data
import matplotlib.pyplot as plt

x = data.listings_host_is_superhost
y = data.scores_checkin

#Scatter plot of listing ratings
plt.rc('font', size=10)
plt.xlabel('Is superhost'); plt.ylabel('Cleanliness rating')
plt.title('Cleanliness rating vs Superhost Status')
plt.rcParams['figure.constrained_layout.use'] = True
plt.scatter(x,y, color='green', s=2)
plt.show()
