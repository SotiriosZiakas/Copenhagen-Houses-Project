import joblib
from build_features import clean_data_transformer
import pandas as pd

# Load the model we created and trained

loaded_model = joblib.load('my_model.pkl')

# Read the new data

new_data = pd.read_csv(r'C:\Users\Sotiris\Desktop\Python\Copenhagen_Houses_Project\houses\src\data\new_house_results.csv')

# Drop the price column (so we suppose that they are unknown)

new_data = new_data.drop('price',axis=1)

# Clean the new data using our custom transformer

new_data = clean_data_transformer.transform(new_data)

# Pass the new data into our pipeline and check the predictions 

predictions = loaded_model.predict(new_data)

print(predictions)