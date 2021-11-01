# Copenhagen Houses Project
==============================

The aim of this project is to build a Machine Learning model, which is trained on house data in the area of Copenhagen and predicts their price. The project is a personal first attempt to use Visual Studio Code and cookiecutter. It is part of my self-study personal projects, where I try new techniques. Working on this project I have learned many new tools and I am very excited to continue and learn even more.


## The steps of the procedure are the following:

1. **train_model:**
    
    a. First we read the data from a **csv** to a **pandas dataframe**.
    
    b. Next we use a custom class imported from *build_features* to clean the data.
    
    c. Then we use a *preprocessor*, which is actually a *ColumnTranformer*. This helps us fill in the missing values using *SimpleImputer* and scale the features with *StandardScaler*.
    
    d. We also treat the categorical feature 'energyClass' using *OrdinalEncoder*.
    
    e. Next step is to create a pipeline, which contains our *preprocessor* and the regressor for this task, which is *RandomForestRegressor*.
    
    f. We split the data using the function *split* whic is also inside *build_features*.
    
    g. Laslty we train the model using our pipeline and save it with the help of **joblib.dump**.

2. **make_predictions:**
    
    a. Load the model trained on the training set using **joblib.load**.
    
    b. Make prediction on the test set.
    
    c. Save and use them to make a plot with real VS predicted prices.

3. **predict_model:**

    a. Load the model trained on the entire dataset using **joblib.load**.
    
    b. Load **new data** that is *unknown* toour model.
    
    c. Make predictions.

4. **visualisations:**

    a. Make a dataframe consisting of the **real** and the **predicted** values, print the **r-squared value** and plot the real VS predicted values using **plotly** for an **interactive visualisation**.
    
    b. Plot **histograms** of the numerical features.
    
    c. Plot the location of each house on the map of Copenhagen based on its **coordinates**.


