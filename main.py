import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle

# Load Data
data = pd.read_csv("train.csv")

# Preprocessing
data['Order Date'] = pd.to_datetime(data['Order Date'], dayfirst=True)
data['Ship Date'] = pd.to_datetime(data['Ship Date'], dayfirst=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region',
                       'Category', 'Sub-Category', 'Product Name']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Select features and target variable
features = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category', 'Postal Code']
X = data[features].fillna(0)
y = data['Sales']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# Save the model
model_file_path = "sales_prediction_xgb_model.pkl"
with open(model_file_path, "wb") as model_file:
    pickle.dump(xgb_model, model_file)

# Load the model
with open(model_file_path, "rb") as model_file:
    model = pickle.load(model_file)

# Define Functions

def predicted_sales_for_input(input_features):
    """Q1: What is the predicted sales for a specific input?"""
    return model.predict(np.array(input_features).reshape(1, -1))[0]

def average_sales_for_region(region, X, y):
    """Q2: What is the average sales for Region X?"""
    region_data = X[X['Region'] == region]
    region_sales = y[region_data.index]
    return np.mean(region_sales)

def total_sales_for_category(category, X, y):
    """Q3: What is the total sales for Category X?"""
    return np.sum(y[X['Category'] == category])

def highest_sales_region(X, y):
    """Q4: Which region has the highest average sales?"""
    regions = X['Region'].unique()
    sales_by_region = {region: np.mean(y[X['Region'] == region]) for region in regions}
    return max(sales_by_region, key=sales_by_region.get)

def trend_for_category(category, X, y):
    """Q5: What is the trend of sales for Category X?"""
    category_data = X[X['Category'] == category]
    category_sales = y[category_data.index]
    return np.mean(category_sales)

def region_sales_distribution(X, y):
    """Q6: What is the sales distribution by region?"""
    regions = X['Region'].unique()
    return {region: np.sum(y[X['Region'] == region]) for region in regions}

def product_sales(product_id, data, y):
    """Q7: What is the total and average sales for Product X?"""
    product_data = data[data['Product Name'] == product_id]
    product_sales = y[product_data.index]
    return {"total": np.sum(product_sales), "average": np.mean(product_sales)}

def predicted_sales_for_city(city, data, y):
    """Q8: What is the predicted average sales for City X?"""
    city_data = data[data['City'] == city]
    city_sales = y[city_data.index]
    return np.mean(city_sales) if len(city_sales) > 0 else "No data for this city"

def total_sales_distribution(X, y):
    """Q9: What is the total sales distribution by category?"""
    categories = X['Category'].unique()
    return {category: np.sum(y[X['Category'] == category]) for category in categories}


def region_sales_comparison_difference(region1, region2, X, y):
    """Q10: What is the difference in sales between Region X and Region Y?"""
    region1_sales = np.mean(y[X['Region'] == region1])
    region2_sales = np.mean(y[X['Region'] == region2])
    return region1_sales - region2_sales

def highest_sales_product(data, y):
    """Q11: What is the product number with the highest sales?"""
    product_sales = data.groupby('Product Name')['Sales'].sum()
    return product_sales.idxmax()

# Generate Answers for Each Question
example_input = X_test.iloc[0].tolist()
answers = {
    "1. Predicted Sales for Input": predicted_sales_for_input(example_input),
    "2. Average Sales for Region 1": average_sales_for_region(1, X, y),
    "3. Total Sales for Category 2": total_sales_for_category(2, X, y),
    "4. Highest Sales Region": highest_sales_region(X, y),
    "5. Sales Trend for Category 2": trend_for_category(2, X, y),
    "6. Sales Distribution by Region": region_sales_distribution(X, y),
    "7. Total and Average Sales for Product 0": product_sales(0, data, y),
    "8. Predicted Sales for City 0": predicted_sales_for_city(0, data, y),
    "9. Total Sales Distribution by Category": total_sales_distribution(X, y),
    "10. Difference in Sales Between Region 1 and Region 2": region_sales_comparison_difference(1, 2, X, y),
    "11. Product with Highest Sales (Product Number)": highest_sales_product(data, y),
}

# Display Answers
for question, answer in answers.items():
    print(f"{question}: {answer}")
