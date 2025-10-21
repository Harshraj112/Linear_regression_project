import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

##import ridge regression and standard scalar pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            
            # Handle categorical variables
            Classes_input = request.form.get('Classes')
            Region_input = request.form.get('Region')
            
            # Encode Classes (adjust these mappings based on your training data)
            classes_mapping = {
                'fire': 1,
                'not fire': 0,
                'no fire': 0
            }
            Classes = classes_mapping.get(Classes_input.lower(), 0)
            
            # Encode Region (adjust these mappings based on your training data)
            region_mapping = {
                'bejaia': 0,
                'sidi bel-abbes': 1,
                'sidibelabbes': 1,
                'region1': 0,
                'region2': 1
            }
            Region = region_mapping.get(Region_input.lower(), 0)

            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge_model.predict(new_data_scaled)

            return render_template('home.html', result=round(result[0], 2))
        
        except ValueError as e:
            error_message = f"Invalid input: Please enter valid numeric values. Error: {str(e)}"
            return render_template('home.html', result=error_message)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('home.html', result=error_message)
    
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(port=5001, debug=True)