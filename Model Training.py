# Train a machine learning model to predict sales
def train_sales_model(processed_data):
    # Feature Selection: Marketing spend, customer count, temperature, etc.
    features = ['Marketing_Spend', 'Customer_Count', 'Temperature', 'Rainfall', 'Holiday_Flag', 'Event_Flag', 'Marketing_Spend_Per_Customer']
    X = processed_data[features]
    y = processed_data['Sales']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gradient Boosting Regressor model
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Save the model for later use
    joblib.dump(model, 'sales_prediction_model.pkl')

    return model

# Train the model using the processed data
model = train_sales_model(processed_data)
