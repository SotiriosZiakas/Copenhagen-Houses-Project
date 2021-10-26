from train_model import X_test
import joblib

# Load the model we created and trained

loaded_model = joblib.load('my_model.pkl')

# Make predictions and save them

predictions = loaded_model.predict(X_test)