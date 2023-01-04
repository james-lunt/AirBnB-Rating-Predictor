import data
import matplotlib.pyplot as plt

def plot_ratings(rating, rating_title):
    #Scatter plot of listing ratings
    plt.rc('font', size=10)
    plt.xlabel('Listings numbered'); plt.ylabel(rating_title + ' rating')
    plt.title('Scatter plot of ' + rating_title + ' ratings')
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(list(range(0, data.num_listings)),rating, color='green', s=2)
    plt.show()

def plot_2d(x,y,xlabel,ylabel,title):
    #Plot against rating
    plt.rc('font', size=10)
    plt.xlabel(ylabel); plt.ylabel(xlabel)
    plt.title(title)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(x,y, color='green', s=2)
    plt.show()

def plot_3d(x1,x2 ,rating, x1label,x2label,ylabel,rating_title):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(x1,x2,rating)
    ax.set_title(rating_title)
    ax.set_xlabel(x1label)
    ax.set_ylabel(x2label)
    ax.set_zlabel(ylabel)
    plt.show()

#plot_threed(data.listings_host_response_rate,data.listings_host_response_time,data.scores_communication, "Communication rating vs Response time & Response rate")
#plot_ratings(data.scores_checkin, "checkin")