import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import json
import joblib

def process_incident_data(csv_path='adani_data.csv'):
    # Load and preprocess data
    df = pd.read_csv(csv_path)

    # Convert date columns to datetime type and extract useful features
    date_columns = ['OCCUREDDATE', 'REPORTEDDATE', 'LASTUPDATEDDATE']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek

    # Create new features from date columns
    df['REPORT_DELAY'] = (df['REPORTEDDATE'] - df['OCCUREDDATE']).dt.total_seconds() / 3600  # in hours

    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['PRIORITY', 'BVCODE', 'BUNAME', 'LMCODE', 'INCIDENTTYPECODE', 'INCIDENTCATCODE', 'SICODE']
    encoders = {}
    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])

    # Create mappings of encoded values to original values
    sicode_mapping = dict(zip(encoders['SICODE'].transform(encoders['SICODE'].classes_),
                              encoders['SICODE'].classes_))
    incidentcatcode_mapping = dict(zip(encoders['INCIDENTCATCODE'].transform(encoders['INCIDENTCATCODE'].classes_),
                                       encoders['INCIDENTCATCODE'].classes_))

    # Split the data into features (X) and target variable (y)
    X = df.drop(['SICODE'] + date_columns, axis=1)
    y = df['SICODE']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions on the entire dataset
    df['predicted_SICODE'] = rf_model.predict(X)

    # Count the occurrences of each predicted SICODE
    sicode_counts = df['predicted_SICODE'].value_counts()

    # Get the top 10 sites with the most incident predictions
    top_10_sites = sicode_counts.head(10)

    # Create a dictionary to store the top 10 sites data
    top_10_sites_data = []
    for site, count in top_10_sites.items():
        original_site = sicode_mapping[site]
        category = incidentcatcode_mapping[df[df['predicted_SICODE'] == site]['INCIDENTCATCODE'].mode()[0]]
        top_10_sites_data.append({
            "siteCode": int(site),
            "siteName": original_site,
            "predictedIncidents": int(count),
            "category": category
        })

    # Convert to JSON
    top_10_sites_json = json.dumps(top_10_sites_data, indent=2)

    # Print the JSON
    print(top_10_sites_json)

    # Export models and encoders for future use
    joblib.dump(rf_model, 'site_code_random_forest_model.joblib')
    joblib.dump(encoders, 'site_code_label_encoders.joblib')
    joblib.dump(sicode_mapping, 'site_code_mapping.joblib')
    joblib.dump(incidentcatcode_mapping, 'site_code_incidentcat_mapping.joblib')

    print("\nModels and encoders exported for future use.")

    return top_10_sites_json

if __name__ == "__main__":
    process_incident_data()