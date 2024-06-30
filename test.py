import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.preprocessing import StandardScaler

# Load the dataset
a1 = pd.read_csv('dataset/case_study1.csv')
a2 = pd.read_csv('dataset/case_study2.csv')

df1 = a1.copy()
df2 = a2.copy()

# Remove nulls
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

columns_to_be_removed = []

for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)

df2 = df2.drop(columns_to_be_removed, axis=1)

for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]

# Merge the two dataframes, inner join so that no nulls are present
df = pd.merge(df1, df2, how='inner', left_on=['PROSPECTID'], right_on=['PROSPECTID'])

# Chi-square test for categorical variables
for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '---', pval)

# VIF for numerical columns
numeric_columns = []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID', 'Approved_Flag']:
        numeric_columns.append(i)

# VIF sequentially check
vif_data = df[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range(total_columns):
    vif_value = variance_inflation_factor(vif_data, column_index)
    print(column_index, '---', vif_value)
    
    if vif_value <= 6:
        columns_to_be_kept.append(numeric_columns[i])
        column_index += 1
    else:
        vif_data = vif_data.drop([numeric_columns[i]], axis=1)

# Check ANOVA for columns_to_be_kept
from scipy.stats import f_oneway
columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])  
    b = list(df['Approved_Flag'])  
    
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']

    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)
        
# Feature selection is done for cat and num features

# Listing all the final features
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]

# Label encoding for the categorical features
df['MARITALSTATUS'].unique()    
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()

# Ordinal feature -- EDUCATION
# SSC            : 1
# 12TH           : 2
# GRADUATE       : 3
# UNDER GRADUATE : 3
# POST-GRADUATE  : 4
# OTHERS         : 1
# PROFESSIONAL   : 3

df.loc[df['EDUCATION'] == 'SSC', 'EDUCATION'] = 1
df.loc[df['EDUCATION'] == '12TH', 'EDUCATION'] = 2
df.loc[df['EDUCATION'] == 'GRADUATE', 'EDUCATION'] = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE', 'EDUCATION'] = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE', 'EDUCATION'] = 4
df.loc[df['EDUCATION'] == 'OTHERS', 'EDUCATION'] = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL', 'EDUCATION'] = 3

df['EDUCATION'] = df['EDUCATION'].astype(int)
df.info()

# One Hot Encoding
df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'], dtype=np.uint8)

# Standard Scaling for numerical columns
columns_to_be_scaled = ['Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment', 
                        'max_recent_level_of_deliq', 'recent_level_of_deliq', 
                        'time_since_recent_enq', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr']

for i in columns_to_be_scaled:
    column_data = df_encoded[i].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_column = scaler.fit_transform(column_data)
    df_encoded[i] = scaled_column

# Saving the dataframe to use in the next step
df_encoded.to_csv('df_encoded.csv', index=False)

# Machine Learning model training and saving
# Features to use for the model
model_features = ['Tot_Missed_Pmnt', 'CC_TL', 'PL_TL', 'Unsecured_TL', 'Age_Oldest_TL',
                  'max_recent_level_of_deliq', 'num_deliq_6_12mts', 'NETMONTHLYINCOME',
                  'Time_With_Curr_Empr', 'CC_enq_L12m']

X = df_encoded[model_features]
y = df_encoded['Approved_Flag']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)
xgb_classifier.fit(X_train, y_train)

# Save the trained model
joblib.dump(xgb_classifier, 'xgb_model.pkl')

# Evaluate the model
y_pred = xgb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

# Save the trained model
joblib.dump(xgb_classifier, r'C:\\Users\\Nikita Rade\\Downloads\\Credit_Risk_Modelling\\xgb_model.pkl')

print("Model trained and saved successfully.")

# Save the label encoder classes
np.save(r'C:\\Users\\Nikita Rade\\Downloads\\Credit_Risk_Modelling\\label_encoder_classes.npy', label_encoder.classes_)


# Save the final df_encoded DataFrame to a CSV file
output_file_path = "C:\\Users\\Nikita Rade\\Downloads\\Credit_Risk_Modelling\\df_encoded.csv"
df_encoded.to_csv(output_file_path, index=False)

print(f"DataFrame saved to {output_file_path}")
