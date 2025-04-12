# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load data from the uploaded file
df = pd.read_csv('Sales_Data_Sample2.csv')

# Display first few rows of the data
df.head()

# Clean and preprocess the data
def preprocess_data(df):
    # Fill missing values with mean for specific columns
    df.fillna({
        'Marketing_Spend': df['Marketing_Spend'].mean(),
        'Temperature': df['Temperature'].mean(),
        'Rainfall': df['Rainfall'].mean(),
        'Customer_Count': df['Customer_Count'].mean()
    }, inplace=True)

    # Convert 'Date' to datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Feature Engineering: Create Day of Week from Date
    df['Day_Of_Week'] = df['Date'].dt.day_name()

    # Grouping data by Store and Date for daily aggregates
    daily_sales = df.groupby(['Store_ID', 'Date']).agg({
        'Sales': 'sum',
        'Marketing_Spend': 'sum',
        'Customer_Count': 'sum',
        'Temperature': 'mean',
        'Rainfall': 'mean',
        'Holiday_Flag': 'mean',
        'Event_Flag': 'mean'
    }).reset_index()

    # Add efficiency metric for marketing spend (spend per customer)
    daily_sales['Marketing_Spend_Per_Customer'] = daily_sales['Marketing_Spend'] / daily_sales['Customer_Count']

    # --- New Analysis with Holiday_Flag and Event_Flag ---
    # Understanding the impact of holidays and events on sales
    # The 'Holiday_Flag' indicates whether the day is a holiday (1 = Holiday, 0 = Non-Holiday)
    # The 'Event_Flag' indicates whether there was a special event on that day (1 = Event, 0 = No Event)

    # Calculate the average sales on holidays and non-holidays
    avg_sales_on_holidays = daily_sales[daily_sales['Holiday_Flag'] == 1]['Sales'].mean()
    avg_sales_on_non_holidays = daily_sales[daily_sales['Holiday_Flag'] == 0]['Sales'].mean()

    # Calculate the average sales on event days and non-event days
    avg_sales_on_events = daily_sales[daily_sales['Event_Flag'] == 1]['Sales'].mean()
    avg_sales_on_non_events = daily_sales[daily_sales['Event_Flag'] == 0]['Sales'].mean()

    print(f"Average sales on holidays: {avg_sales_on_holidays}")
    print(f"Average sales on non-holidays: {avg_sales_on_non_holidays}")
    print(f"Average sales on event days: {avg_sales_on_events}")
    print(f"Average sales on non-event days: {avg_sales_on_non_events}")

    # Return the processed data with all features
    return daily_sales

# Preprocess the data
processed_data = preprocess_data(df)

# Display the processed data
processed_data.head()
