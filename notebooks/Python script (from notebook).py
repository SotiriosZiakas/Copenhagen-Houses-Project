# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# %%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# %%
# Read the data from a csv file to a pandas dataframe and check the head of it

my_data = pd.read_csv(r'C:\Users\Sotiris\Desktop\Python\Copenhagen_Houses_Project\houses\src\data\house_results.csv')
my_data.head()


# %%
my_data.info()


# %%
my_data.describe()


# %%
# Create a function which will help us clean our data

def clean_data(df):

    # Make a copy of the original dataframe
    df_copy = df.copy()

    # Drop the unnecessary columns

    df_copy = df_copy.drop(['id','city','street','squaremeterPrice','priceChangePercentTotal'],axis=1)

    # Drop the null energy classes

    df_copy = df_copy.dropna(subset=['energyClass'])

    # Drop the unknown energy classes
    df_copy = df_copy[df_copy['energyClass'] != '-']

    # Correct the energy classes
    replace_dict = {'A':'A20','I':'A20','J':'A15','K':'A10','L':'A15','M':'A20'}
    df_copy['energyClass'] = df_copy['energyClass'].str.upper()
    df_copy['energyClass'] = df_copy['energyClass'].replace(replace_dict)

    # Drop the houses with build year 0
    df_copy = df_copy[df_copy['buildYear'] != 0]

    # Drop the houses with over 1000 square meters
    df_copy = df_copy[df_copy['size'] < 1000]

    # Return the transformed dataframe
    return df_copy


# %%
# Convert the custom function into a transformer

clean_data_transformer = FunctionTransformer(clean_data)

# Transform/clean the data

my_data = clean_data_transformer.transform(my_data)


# %%
# Histograms of the numerical data

my_data.hist(bins= 30,figsize=(22,12));


# %%
# Save the min and max of the coordinates in our cleaned dataframe

long_max = my_data['longitude'].max()
long_min = my_data['longitude'].min()
lat_max = my_data['latitude'].max()
lat_min = my_data['latitude'].min()

# Make a box of coordinates

BBox = ((long_min,long_max,lat_min,lat_max))

# Read the map of Copenhagen

path = 'C:/Users/Sotiris/Desktop/cph_map.png'
img = plt.imread(path)

# Create the plot

fig, ax = plt.subplots(figsize = (12,12))
sns.scatterplot(x='longitude',y='latitude',data=my_data,alpha=0.4,palette='jet',hue='price')
ax.set_title('Plotting Spatial Data on Copenhagen Map')
ax.set_xlim(long_min,long_max) # Set the x-axis to be between long_min and long_max
ax.set_ylim(lat_min,lat_max) # Set the y-axis to be between lat_min and lat_max
ax.imshow(img, extent = BBox, aspect= 'equal');


# %%
# Split the data using train/test split

X = my_data.drop('price',axis=1)
y = my_data['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=42)


# %%
# Create a preprocessor consisting of numerical and categorical transformers and make a pipeline

numeric_features = ['latitude','longitude','propertyType','rooms','size','lotSize','floor','buildYear','zipCode','basementSize']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant',missing_values = np.nan, fill_value = 0)),
    ('scaler', StandardScaler())
])

categorical_features = ['energyClass']
categorical_transformer = Pipeline(steps=[
    ('ordinal_encoder', OrdinalEncoder(categories=[['A20','A15','A10','B','C','D','E','F','G']]))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

reg = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor())
])

# Fit the model and make predictions

reg.fit(X_train,y_train)
predictions = reg.predict(X_test)

# Compute the score, which is the r-squared value for this particular model

reg.score(X_test,y_test)


# %%
# Make a dataframe consisting of both real and predicted values

df1 = pd.DataFrame(predictions,columns=['Predicted_Prices'])
test = y_test.reset_index()
test = test.rename(columns={'price':'Real_Prices'})
df2 = test.drop('index',axis=1)
df = df1.join(df2)

# Plot real VS predicted values

px.scatter(df)


