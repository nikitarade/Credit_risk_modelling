from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the trained XGBoost model
model_path = 'xgb_model.pkl'
xgb_classifier = joblib.load(model_path)

# Load the preprocessed dataframe for feature scaling
df_encoded_path = 'df_encoded.csv'
df_encoded = pd.read_csv(df_encoded_path)

# Extract feature columns used for training the model
model_features = [
    'Tot_Missed_Pmnt', 'CC_TL', 'PL_TL', 'Unsecured_TL', 'Age_Oldest_TL',
    'max_recent_level_of_deliq', 'num_deliq_6_12mts', 'NETMONTHLYINCOME',
    'Time_With_Curr_Empr', 'CC_enq_L12m'
]

# Load the standard scaler used for feature scaling
scaler = StandardScaler()
scaler.fit(df_encoded[model_features])

# Load the label encoder used for the target variable encoding
label_encoder_classes = np.load(r'label_encoder_classes.npy', allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    form_data = {
        'Tot_Missed_Pmnt': float(request.form['Tot_Missed_Pmnt']),
        'CC_TL': float(request.form['CC_TL']),
        'PL_TL': float(request.form['PL_TL']),
        'Unsecured_TL': float(request.form['Unsecured_TL']),
        'Age_Oldest_TL': float(request.form['Age_Oldest_TL']),
        'max_recent_level_of_deliq': float(request.form['max_recent_level_of_deliq']),
        'num_deliq_6_12mts': float(request.form['num_deliq_6_12mts']),
        'NETMONTHLYINCOME': float(request.form['NETMONTHLYINCOME']),
        'Time_With_Curr_Empr': float(request.form['Time_With_Curr_Empr']),
        'CC_enq_L12m': float(request.form['CC_enq_L12m']),
        'MARITALSTATUS': request.form['MARITALSTATUS'],
        'EDUCATION': request.form['EDUCATION'],
        'GENDER': request.form['GENDER'],
        'last_prod_enq2': request.form['last_prod_enq2'],
        'first_prod_enq2': request.form['first_prod_enq2']
    }

    # Create a DataFrame from the form data
    input_df = pd.DataFrame([form_data])

    # One-Hot Encode the categorical features
    input_df_encoded = pd.get_dummies(input_df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2', 'EDUCATION'], dtype=np.uint8)

    # Add missing columns with default value 0
    for col in df_encoded.columns:
        if col not in input_df_encoded.columns and col not in ['Approved_Flag']:
            input_df_encoded[col] = 0

    # Ensure the columns are in the same order as during training
    input_df_encoded = input_df_encoded[model_features + list(input_df_encoded.columns.difference(model_features))]

    # Standardize the features
    input_df_encoded[model_features] = scaler.transform(input_df_encoded[model_features])

    # Ensure the feature names match
    model_feature_names = xgb_classifier.get_booster().feature_names
    input_df_encoded = input_df_encoded[model_feature_names]

    # Print the feature names to debug any mismatches
    print("Model feature names:", model_feature_names)
    print("Input DataFrame feature names:", list(input_df_encoded.columns))

    # Predict the class probabilities
    prediction = xgb_classifier.predict(input_df_encoded)
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    return render_template('index.html', prediction_text=f'Predicted class: {predicted_class}')

if __name__ == '__main__':
    app.run(debug=True)
