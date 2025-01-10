from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Path to the saved model file
model_file_path = 'C:/Users/parth/Desktop/Third task max/linear_regression_model.pkl'

# Check if the file exists
if os.path.exists(model_file_path):
    # Load the trained model
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)
else:
    print(f"Error: Model file not found at {model_file_path}")
    model = None

# Route for rendering the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Check if the model is loaded properly
            if model is None:
                return render_template('index.html', prediction_text="Model not found. Please check the model file.")

            # Get values from the form (assumes the form fields match the input names)
            square_feet = float(request.form['square_feet'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            neighborhood = int(request.form['neighborhood'])  # 0 for Rural, 1 for Suburb, 2 for Urban
            year_built = int(request.form['year_built'])

            # Prepare the input data for prediction
            input_features = np.array([[square_feet, bedrooms, bathrooms, neighborhood, year_built]])

            # Make the prediction using the trained model
            prediction = model.predict(input_features)

            # Return the result to the frontend
            return render_template('index.html', prediction_text=f'Predicted Price: ${prediction[0]:,.2f}')
        
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
