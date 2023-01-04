import numpy as np
import pandas as pd
import math

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
listings_neighbourhood_cleansed = df.iloc[:,28] #Label string e.g. 'Dun Laoghaire'
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

def make_set(feature):
    set_of_something = set()
    for i in feature:
        set_of_something.add(i)
    return set_of_something, len(set_of_something)

#Reviews
df = pd.read_csv('reviews.csv', sep=',')
reviews_listing_id = df.iloc[:,0]
reviews_date = df.iloc[:,2]
reviews_comments = df.iloc[:,5]

#Takes list of inputs and output
def pop_nans(inputs,outputs):
    for i in outputs.keys():
        if math.isnan(outputs[i]):
            for input in inputs:
                input.pop(i)
            outputs.pop(i)

    for input in inputs:
        for i in input.keys():
            try:
                if math.isnan(input[i]):
                    for input1 in inputs:
                        input1.pop(i)
                    outputs.pop(i)
            except:
                continue
    return inputs,outputs

def pop_nans_y(inputs,outputs):
    for i in outputs.keys():
        if math.isnan(outputs[i]):
            for input in inputs:
                input.pop(i)
            outputs.pop(i)
    return inputs,outputs


            
