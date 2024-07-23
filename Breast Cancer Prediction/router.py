# Import the Libraries
import pandas as pd
from flask import Flask, redirect, render_template, request
import joblib
import os
# the function I craeted to process the data in utils.py
from utils import preprocess_new


# Intialize the Flask APP
app = Flask(__name__)

# Loading the Model
model = joblib.load('model_RandomForest.pkl')

# # Route for Home page


@app.route('/')
def home():
    return render_template('index.html')

# Route for Predict page

 
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':  # while prediction # POST mean that the user enter and wait to GET
        ## the arguments, radiues_mean... should be the same as the columns in the data.
        ## the appreviations in ' ' are the same as in the which in the name attribute in the predict file.
        radius_mean = float(request.form['rm'])
        texture_mean = float(request.form['tm'])
        perimeter_mean = float(request.form['pm'])
        area_mean  = float(request.form['am'])
        smoothness_mean = float(request.form['sm'])
        compactness_mean = float(request.form['cm'])
        concavity_mean = float(request.form['cmn'])
        concave_points_mean = float(request.form['cpm'])
        symmetry_mean = float(request.form['smm'])
        fractal_dimension_mean = float(request.form['fdm'])
        radius_se = float(request.form['rs'])
        texture_se = float(request.form['ts'])
        perimeter_se = float(request.form['ps'])
        area_se = float(request.form['ase'])
        smoothness_se = float(request.form['sse'])
        compactness_se = float(request.form['cse'])
        concavity_se = float(request.form['cs'])
        concave_points_se = float(request.form['cps'])
        symmetry_se = float(request.form['ss'])
        fractal_dimension_se = float(request.form['fds'])
        radius_worst = float(request.form['rw'])
        texture_worst = float(request.form['tw'])
        perimeter_worst = float(request.form['pw'])
        area_worst = float(request.form['aw'])
        smoothness_worst = float(request.form['sow'])
        compactness_worst = float(request.form['cow'])
        concavity_worst = float(request.form['cw'])
        concave_points_worst = float(request.form['cpw'])
        symmetry_worst = float(request.form['sw'])
        fractal_dimension_worst = float(request.form['fdw'])
        
        # Concatenate all Inputs
        # Here we take the input from POSt request and put it in data frame
        X_new = pd.DataFrame({
        'radius_mean': [radius_mean],
        'texture_mean': [texture_mean],
        'perimeter_mean': [perimeter_mean],
        'area_mean': [area_mean],
        'smoothness_mean': [smoothness_mean],
        'compactness_mean': [compactness_mean],
        'concavity_mean': [concavity_mean],
        'concave points_mean': [concave_points_mean],
        'symmetry_mean': [symmetry_mean],
        'fractal_dimension_mean': [fractal_dimension_mean],
        'radius_se': [radius_se],
        'texture_se': [texture_se],
        'perimeter_se': [perimeter_se],
        'area_se': [area_se],
        'smoothness_se': [smoothness_se],
        'compactness_se': [compactness_se],
        'concavity_se': [concavity_se],
        'concave points_se': [concave_points_se],
        'symmetry_se': [symmetry_se],
        'fractal_dimension_se': [fractal_dimension_se],
        'radius_worst': [radius_worst],
        'texture_worst': [texture_worst],
        'perimeter_worst': [perimeter_worst],
        'area_worst': [area_worst],
        'smoothness_worst': [smoothness_worst],
        'compactness_worst': [compactness_worst],
        'concavity_worst': [concavity_worst],
        'concave points_worst': [concave_points_worst],
        'symmetry_worst': [symmetry_worst],
        'fractal_dimension_worst': [fractal_dimension_worst]
        })

        # Call the Function and Preprocess the New Instances
        X_processed = preprocess_new(X_new)

        # call the Model and predict
        y_pred_new = model.predict(X_processed)
        # we do this step for numerical prediction in regression problems
        #y_pred_new = '{:.4f}'.format(y_pred_new[0])
        print(y_pred_new)
        return render_template('predict.html', pred_val=y_pred_new)
    else:
        print(request.form)
        return render_template('predict.html')


# Route for About page
@app.route('/about')
def about():
    return render_template('about.html')


# Run the App from the Terminal
if __name__ == '__main__':
    app.run(debug=True)