""" Read the csv file into a pandas Dataframe """

import pandas as pd

data = pd.read_csv(r"C:\Users\Sotiris\Desktop\Python\Copenhagen_Houses_Project\houses\src\data\house_results.csv")


""" Clean the Dataframe using our custom transformer """

# Import the transformer clean_data_transformer from build_features

from build_features import clean_data_transformer

my_data = clean_data_transformer.transform(data)

""" 
Build a preprocessor which uses SimpleImputer and OrdinalEncoder
to transform numerical and categorical features

"""

from build_features import build_preprocessor

preprocessor = build_preprocessor()

"""
Make a 2 step pipeline, in which first impute missing values of the floor column & order the energy levels using our preprocessor
and then we pass the result to our regressor model

"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

reg = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor())
])



""" Split the data """

# Import the function split from split_data

from build_features import split

X_train, X_test, y_train, y_test = split(my_data)


""" 

Fit the model using the pipeline we made
and save it using joblib.dump

"""

reg.fit(X_train,y_train)

import joblib

joblib.dump(reg,'my_model.pkl')


""" Train the model using the entire dataset so we do not throw away potentially valuable data """

X = my_data.drop('price',axis=1)
y = my_data['price']

reg.fit(X,y)

# Save the model that is trained on the entire dataset

joblib.dump(reg,'full_trained_model.pkl')