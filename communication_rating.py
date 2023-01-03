import data
import ratings_plot

x1 = data.listings_host_response_rate
x2 = data.listings_host_response_time
y = data.scores_communication
print(len(x1))
print(len(x2))

x,y = data.pop_nans([x1,x2],y)
#x,y = data.pop_nans_x([x[0].to_numpy(),x[1].to_numpy()],y)
x1 = x[0]
x2 = x[1]
print(len(x1))
print(len(x2))

set1,set1length = data.make_set(x1)
set2,set2length = data.make_set(x2)

#convert percentage to float
for i in x1.keys():
    x1[i] = float(x1[i].strip("%"))

#Map rate to int
for i in x2.keys():
    if x2[i] == "within an hour":
        x2[i] = 4
    if x2[i] == "within a few hours":
        x2[i] = 3
    if x2[i] == "within a day":
        x2[i] = 2
    if x2[i] == "a few days or more":
        x2[i] = 1

print(x1)

ratings_plot.plot_threed(x1,x2,y, "Communication rating vs Response time & Response rate")
#plot_ratings(data.scores_checkin, "checkin")