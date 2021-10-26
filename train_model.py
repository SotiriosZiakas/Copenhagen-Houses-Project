""" Read the csv file into a pandas Dataframe """

# Import the class drop_correct from build_features

from build_features import drop_correct

import pandas as pd

data = pd.read_csv(r"C:\Users\Sotiris\Desktop\Python\Copenhagen_Houses_Project\houses\src\data\house_results.csv")


""" Clean the Dataframe """

# Create an instance of the class and clean the data

drop_cor = drop_correct()
my_data = drop_cor.transform(data)

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



""" Split the data"""

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