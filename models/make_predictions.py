""" Make predictions on the test set """

import joblib
from train_model import X_test

# Load the model that we trained
trained_model = joblib.load('my_model.pkl')

# Make predictions on the test set
predictions = trained_model.predict(X_test)
