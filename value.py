import data
import numpy as np
import ratings_plot
import matplotlib.pyplot as plt

inputs = [data.listings_amenities,data.listings_accommodates,data.listings_bedrooms,data.listings_beds,data.listings_price]
inputs,outputs = data.pop_nans(inputs,data.scores_value)

#convert money to float
for i in inputs[4].keys():
    inputs[4][i] = float(inputs[4][i].strip("$").replace(',', ''))
    if inputs[4][i] > 2000:
        inputs[4][i] = 181.8111094043352 #mean

#Use number of amenities instead of amenities themselves
for i in inputs[0].keys():
    inputs[0][i] = inputs[0][i].count(',')

ratings_plot.plot_3d(inputs[2],inputs[4],outputs,"#Bedrooms","Price","Value","Value Rating vs Price & #Bedrooms")

X=np.column_stack((inputs[0]*inputs[4],inputs[1]*inputs[4],inputs[2]*inputs[4],inputs[3]*inputs[4]))

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, outputs)

print(model.intercept_)
print(model.coef_)