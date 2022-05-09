"""" Function which splits the data using train/test split """

from sklearn.model_selection import train_test_split

def split(df):

    X = df.drop('price',axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=42)

    return X_train, X_test, y_train, y_test


"""
Function which takes a copy of a Dataframe
drops some columns that are not useful, 
then drops the houses with null and unknown energy class and corrects the remaining energy classes.
Finally drops the houses with build year equal to zero and the ones with size greater than 1000 square meters
"""

from sklearn.preprocessing import FunctionTransformer

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

# Convert the custom function into a transformer

clean_data_transformer = FunctionTransformer(clean_data)




""" Define the preprocessor """

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_preprocessor():

    # Fill the empty values with zeros and scale the numerical features

    numeric_features = ['latitude','longitude','propertyType','rooms','size','lotSize','floor','buildYear','zipCode','basementSize']
    numeric_transformer = Pipeline(steps=[
        ('imputer' , SimpleImputer(strategy='constant',missing_values = np.nan, fill_value = 0)),
        ('scaler', StandardScaler())
            ])

    # Put the energy labels in order from A20 to G (best to worst)

    categorical_features = ['energyClass']
    categorical_transformer = Pipeline(steps=[
        ('ordinal_encoder', OrdinalEncoder(categories=[['A20','A15','A10','B','C','D','E','F','G']]))
            ])
    

    # Combine the above two procedures in one

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
            ])
    return preprocessor
