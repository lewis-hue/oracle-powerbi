# Generate predictions and prepare data for Power BI visualization
def generate_predictions_for_power_bi(processed_data, model):
    # Use the model to make predictions on the entire dataset
    features = ['Marketing_Spend', 'Customer_Count', 'Temperature', 'Rainfall', 'Holiday_Flag', 'Event_Flag', 'Marketing_Spend_Per_Customer']
    processed_data['Predicted_Sales'] = model.predict(processed_data[features])

    # Calculate Marketing ROI (Predicted Sales / Marketing Spend)
    processed_data['Marketing_ROI'] = processed_data['Predicted_Sales'] / processed_data['Marketing_Spend']

    # Save the results for Power BI consumption
    processed_data.to_csv('sales_predictions_for_power_bi2.csv', index=False)

    return processed_data

# Generate predictions and save them to CSV
predicted_data = generate_predictions_for_power_bi(processed_data, model)

# Display the predicted data
predicted_data.head()
