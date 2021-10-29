# Copenhagen Houses Project


The aim of this project is to build a Machine Learning model, which is trained on house data in the area of Copenhagen and predicts their price. The project is a personal first attempt to use Visual Studio Code and cookiecutter. It is part of my self-study personal projects, where I try new techniques. Working on this project I have learned many new tools and I am very excited to continue and learn even more.


## The steps of the procedure are the following:

1. **train_model:**
    
    1. First we read the data from a **csv** file to a **pandas dataframe**.
    2. Next we use a custom transformer imported from *build_features* to clean the data. 
    3. Then we use a *preprocessor*, which is actually a *ColumnTranformer*. This helps us fill in the missing values using *SimpleImputer* and scale the features with *StandardScaler*.
    4. We also treat the categorical feature 'energyClass' using *OrdinalEncoder*.
    5. Next step is to create a pipeline, which contains our *preprocessor* and the regressor for this task, which is *RandomForestRegressor*.
    6. We split the data using the function *split* which is also inside *build_features*.
    7. Laslty we train the model using our pipeline and save it with the help of **joblib.dump**.

2. **predict_model:**

    1. Load the model we created using **joblib.load**.
    2. Read **new data** that is unknown to our model.
    3. Make predictions.

3. **visualizations:**

    1. Make a dataframe consisting of the **real** and the **predicted** values, print the **r-squared value** and plot the real VS predicted values using **plotly** for an **interactive visualization**.
    2. Plot **histograms** of the numerical features.
    3. Plot the location of each house on the map of Copenhagen based on its **coordinates**.


