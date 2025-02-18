import pickle
import json
from flask import Flask, render_template, request
import numpy as np

# Load the model and columns JSON
model = pickle.load(open('model/banglore_home_prices_model.pickle', 'rb'))
with open('model/columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# Remove unwanted values (the first three columns)
locations = [column for column in data_columns if column not in ['sqft', 'bath', 'bhk', 'total_sqft']]

# Flask setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', columns=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    # Prepare the input array
    loc_index = data_columns.index(location.lower()) if location in data_columns else -1
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    # Predict the price
    price = model.predict([x])[0]
    price_in_lakhs = price / 10  # Converting to lakhs

    # Return the result in lakhs
    return render_template('index.html', prediction=round(price_in_lakhs, 2), columns=locations)

if __name__ == "__main__":
    app.run(debug=True)
