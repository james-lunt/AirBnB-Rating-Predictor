import data
import statistics
from collections import defaultdict
import numpy as np
import pandas as pd
np.set_printoptions(threshold=None)
import math

#Need to pop all inputs with no review score
x, y = data.pop_nans_y([data.listings_number_of_reviews,data.listings_host_is_superhost, data.listings_neighbourhood,
                      data.listings_amenities,data.listings_accommodates,data.listings_bedrooms,
                      data.listings_beds,data.listings_price],data.scores_rating)


def normalise_values(data_points):
    for key in data_points.keys():
        data_points[key] = (data_points[key] - min(data_points)) / (max(data_points) - min(data_points))
    return data_points

#Number of reviews for weighting
#Also gets rid of inputs with 0 reviews
reviews_num = x[0].astype("object")
reviews_num = normalise_values(reviews_num)

#Superhost statuses
is_superhost = x[1]
for key in is_superhost.keys():
    if is_superhost[key] == "t":
        is_superhost[key] = 1
    else:
        is_superhost[key] = 0

#Calculate sentiment of neigbourhood by averaging location ratings
#Use neighbourhoods cleansed as neighbourhood feature vector is missing alot of features
#neighbourhoods, location_ratings = data.pop_nans([data.listings_neighbourhood_cleansed],data.scores_location)
location_ratings = data.scores_location
neighbourhoods = x[2]
#Replace NaNs with cleansed location as cleansed is never NaN
neighbourhoods_cleansed = data.listings_neighbourhood_cleansed
for key in neighbourhoods.keys():
    #print(neighbourhoods[key])
    try:
        if math.isnan(neighbourhoods[key]):
            neighbourhoods[key] = neighbourhoods_cleansed[key]
    except:
        continue

neighbourhood_ranking = defaultdict(list)

for key in neighbourhoods.keys():
    if not math.isnan(location_ratings[key]):
        neighbourhood_ranking[neighbourhoods[key]].append(location_ratings[key])

for neighbourhood, ratings in neighbourhood_ranking.items():
    av_rating = statistics.mean(ratings)
    neighbourhood_ranking[neighbourhood] = av_rating
#print(neighbourhood_ranking)

for key in neighbourhood_ranking.keys():
    neighbourhood_ranking[key] = (neighbourhood_ranking[key] - min(neighbourhood_ranking.values())) / (max(neighbourhood_ranking.values()) - min(neighbourhood_ranking.values()))

#Change neighbourhood value to sentiment
for i in neighbourhood_ranking.keys():
    for j in neighbourhoods.keys():
        if i == neighbourhoods[j]:
            neighbourhoods[j] = neighbourhood_ranking[i]

#A few outliers due to mispelling or something? Set to min value
for i in neighbourhoods.keys():
    if isinstance(neighbourhoods[i],str):
        neighbourhoods[i] = min(neighbourhood_ranking.values())
#Unseen Item will be assigned ranking given the neighbourhood

#Value based on combined metrics
value_inputs = [x[3],x[4],x[5],x[6],x[7]]
#value_inputs,outputs = data.pop_nans(value_inputs,data.scores_value)
outputs = data.scores_value

#convert money to float
for i in value_inputs[4].keys():
    value_inputs[4][i] = float(value_inputs[4][i].strip("$").replace(',', ''))

#Use number of amenities instead of amenities themselves
for i in value_inputs[0].keys():
    value_inputs[0][i] = value_inputs[0][i].count(',')

#Convert N/A bedrooms to 1.5 the number it accommodates
for key in value_inputs[3].keys():
    if math.isnan(value_inputs[3][key]):
        value_inputs[3][key] = value_inputs[1][key]/1.5

#Convert N/A beds to 1.5 the number of bedrooms
for key in value_inputs[2].keys():
    if math.isnan(value_inputs[2][key]):
        value_inputs[2][key] = value_inputs[3][key]/1.5


#If price, value will be smaller
value_sentiment = (value_inputs[0] + value_inputs[1] + value_inputs[2] + value_inputs[3])/value_inputs[4]
value_sentiment = normalise_values(value_sentiment)

print(value_sentiment)
print(is_superhost)
print(reviews_num)
print(neighbourhoods)

X=np.column_stack((is_superhost*reviews_num,value_sentiment*reviews_num,neighbourhoods*reviews_num))
#X = np.array([[is_superhost*reviews_num],[ranked_neighbourhoods*reviews_num],[value_sentiment*reviews_num]])


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

print(model.intercept_)
print(model.coef_)

"""
    for j in neighbourhoods.keys():
        if i == neighbourhoods[j]:
            neighbourhoods[j] = neighbourhood_ranking[i]"
"""