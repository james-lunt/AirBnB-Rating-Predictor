import data
import matplotlib.pyplot as plt
import statistics

y = data.scores_communication
x, y = data.pop_nans([data.listings_host_is_superhost],y)
x = x[0]
x = x.to_numpy()
y = y.to_numpy()

superhost = 0
nonsuperhost = 0
superhost_scores = []
nonsuperhost_scores = []
for i in range (len(x)):
    if x[i] == 't':
        superhost+=1
        superhost_scores.append(y[i])
    else:
        nonsuperhost+=1
        nonsuperhost_scores.append(y[i])

print(len(x))
print(superhost_scores)

print(statistics.stdev(superhost_scores))
print(statistics.stdev(nonsuperhost_scores))

#Scatter plot of listing ratings
plt.rc('font', size=10)
plt.xlabel('Is superhost'); plt.ylabel('Communication rating')
plt.title('Communication rating vs Superhost Status')
plt.rcParams['figure.constrained_layout.use'] = True
plt.scatter(x,y, color='green', s=2)
plt.show()
