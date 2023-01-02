import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv('listings.csv', sep=',')
#Scores Outputs
scores_rating = df.iloc[:,61]
scores_accuracy = df.iloc[:,62]
scores_cleanliness = df.iloc[:,63]
scores_checkin = df.iloc[:,64]
scores_communication = df.iloc[:,65]
scores_location = df.iloc[:,66]
scores_value = df.iloc[:,67]

#Listings ID
listings_id = df.iloc[:,0]
#Number of listings
num_listings = len(listings_id)

#Features
listing_description = df.iloc[:,6] #String
listings_neighborhood_description = df.iloc[:,7] #String
listings_host_since = df.iloc[:,12] #Date e.g. 06/08/2010
listings_host_location = df.iloc[:,13] #Label string e.g. 'Dublin,Ireland'
listings_host_response_time = df.iloc[:,15] #Label string e.g, 'Within an hour'
listings_host_response_rate = df.iloc[:,16] #Percentage
listings_host_is_superhost = df.iloc[:,18] #Boolean probably read as a String either 't' or 'f'
listings_neighbourhood = df.iloc[:,27] #Label string e.g. 'Dun Laoghaire'
listings_longitude = df.iloc[:,30] #Longitude Latitude tuple e.g. (53.29178, -6.25792)
listings_latitude = df.iloc[:,31] 
listings_property_type = df.iloc[:,32] #Label string e.g. 'Private room in bungalow'
listings_room_type = df.iloc[:,33] #Label string e.g. 'Entire home/apt'
listings_accommodates = df.iloc[:,34] #Integer indicate number of people
listings_bathroom_text = df.iloc[:,36] #Label string e.g. '1.5 shared baths'
listings_bedrooms = df.iloc[:,37] #Integer indicate number of bedrooms
listings_beds = df.iloc[:,38] #Integer indicate number of beds
listings_amenities = df.iloc[:,39] #List of strings e.g. ["Oven", "Free Parking", "Shower"]
listings_price = df.iloc[:,40] #Dollars $
listings_number_of_reviews = df.iloc[:,56] #Integer indicating the number of reviews written for the listing


#Reviews
df = pd.read_csv('reviews.csv', sep=',')
reviews_listing_id = df.iloc[:,0]
reviews_date = df.iloc[:,2]
reviews_comments = df.iloc[:,5]

#kNN Regression model for location scores
x1 = listings_longitude
x2 = listings_latitude
y = scores_location
for i in range(len(listings_number_of_reviews)):
    if listings_number_of_reviews[i] == 0:
        x1.pop(i)
        x2.pop(i)
        y.pop(i)

#y = y.to_numpy()
X=np.column_stack((x1,x2))

"""
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],y)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()
"""
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=True)
print("Hi")
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X,y)
#Prediction
ypred = model.predict(X)
print(ypred)
print(y)

#Scatter plot of listing ratings
"""plt.rc('font', size=10)
plt.xlabel('Listings numbered'); plt.ylabel('Score rating')
plt.title('Scatter plot of listing ratings')
plt.rcParams['figure.constrained_layout.use'] = True
plt.scatter(list(range(0, num_listings)),scores_rating, color='green', s=2)
plt.show()"""



