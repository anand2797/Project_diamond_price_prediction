from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler


app = Flask(__name__)

# Load the trained model
model = joblib.load('model_rfr.pkl')

# Define the column names
column_names = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']

# Load the dataset
dataset = pd.read_csv('diamonds_cleaned_data.csv')

# Get the unique values for categorical columns
cut_values = dataset['cut'].unique()
color_values = dataset['color'].unique()
clarity_values = dataset['clarity'].unique()

# Fit the encoders on the dataset
ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(dataset[['cut']])

onehot_encoder = OneHotEncoder(drop='first')
onehot_encoder.fit(dataset[['color', 'clarity']])


@app.route('/')
def home():
    return render_template('index.html',
                           cut_values=cut_values, color_values=color_values, clarity_values=clarity_values)


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the form
    carat = float(request.form.get('carat'))
    cut = request.form.get('cut')
    color = request.form.get('color')
    clarity = request.form.get('clarity')
    depth = float(request.form.get('depth')) 
    table = float(request.form.get('table'))
    x = float(request.form.get('x'))
    y = float(request.form.get('y'))
    z = float(request.form.get('z'))

    # Create a DataFrame with the input data
    input_df = pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]], columns=column_names)

    # Encode the categorical columns
    ordinal_encoded = ordinal_encoder.transform(input_df[['cut']])
    onehot_encoded = onehot_encoder.transform(input_df[['color', 'clarity']])

    # Apply log transformation to numerical columns
    numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
    input_df[numerical_columns] = np.log(input_df[numerical_columns])
    
    # Perform scaling on numerical columns
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(input_df[numerical_columns])

    # Concatenate the preprocessed data
    input_processed = np.concatenate((onehot_encoded.toarray(), ordinal_encoded, numerical_scaled), axis=1)

    # Make the price prediction
    price = model.predict(input_processed)

    # Render the result template with the predicted price
    return render_template('index.html', price=price[0], cut_values=cut_values, color_values=color_values,
                           clarity_values=clarity_values, carat=carat, cut=cut, color=color, clarity=clarity,
                           depth=depth, table=table, x=x, y=y, z=z)


if __name__ == '__main__':
    app.run(debug=True)
