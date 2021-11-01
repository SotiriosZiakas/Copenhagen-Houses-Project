""" Function for plotting spatial data on Copenhagen map"""
#%%
import matplotlib.pyplot as plt
import seaborn as sns

from train_model import my_data 

from sklearn.metrics import r2_score
from train_model import y_test
from make_predictions import predictions
import plotly.express as px
import pandas as pd

""" Plot predicted VS real prices using plotly """

# Make a dataframe consisting of real and predicted values

df1 = pd.DataFrame(predictions,columns=['Predicted_Prices'])
test = y_test.reset_index()
test = test.rename(columns={'price':'Real_Prices'})
df2 = test.drop('index',axis=1)
df = df1.join(df2)

fig = px.scatter(df)
fig.show();

""" Calculate and print the r-squared value """

result = r2_score(y_test,predictions)
print(result)


# %%

# Histograms of the numerical features

my_data.hist(bins= 30,figsize=(22,12));

#%%

# Find the maximum and minimum values for longitude and latitude of a given dataframe

long_max = my_data['longitude'].max()
long_min = my_data['longitude'].min()
lat_max = my_data['latitude'].max()
lat_min = my_data['latitude'].min()

# Make a bounding box using the above variables

BBox = ((long_min,long_max,lat_min,lat_max))
    
# Read the image

img = plt.imread(r'C:\Users\Sotiris\Desktop\Python\Copenhagen_Houses_Project\houses\src\data\Copenhagen_map.png')

# Plot the location of each house on the map based on its coordinates

fig, ax = plt.subplots(figsize = (12,12))
sns.scatterplot(x='longitude',y='latitude',data=my_data,alpha=0.4,palette='jet',hue='price')
ax.set_title('Plotting Spatial Data on Copenhagen Map')

# Set the x-axis to be between long_min and long_max
ax.set_xlim(long_min,long_max) 

# Set the y-axis to be between lat_min and lat_max
ax.set_ylim(lat_min,lat_max) 

ax.imshow(img, extent = BBox, aspect= 'equal');