import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IncidentTypePredictor:
    def __init__(self):
        self.rf_model = None
        self.le = None
        self.feature_columns = None

    def train_model(self, df):
        logging.info("Starting model training for INCIDENTTYPECODE")
        # Data Preparation and Feature Engineering
        date_columns = ['REPORTEDDATE', 'LASTUPDATEDDATE', 'OCCUREDDATE']
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')

        df['Day_of_Week'] = df['OCCUREDDATE'].dt.dayofweek
        df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)

        categorical_cols = ['PRIORITY', 'BVCODE', 'BUNAME', 'SICODE', 'LMCODE', 'INCIDENTCATCODE']
        df_encoded = pd.get_dummies(df, columns=categorical_cols)

        self.le = LabelEncoder()
        df_encoded['INCIDENTTYPECODE'] = self.le.fit_transform(df['INCIDENTTYPECODE'])

        X = df_encoded.drop(['INCIDENTTYPECODE', 'REPORTEDDATE', 'LASTUPDATEDDATE', 'OCCUREDDATE'], axis=1)
        y = df_encoded['INCIDENTTYPECODE']

        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)

        self.feature_columns = list(X.columns)

        # Save the model, label encoder, and feature columns
        joblib.dump(self.rf_model, 'incident_type_model.joblib')
        joblib.dump(self.le, 'incident_type_label_encoder.joblib')
        joblib.dump(self.feature_columns, 'incident_type_features.joblib')
        logging.info("Model training completed and saved")

    def get_incidenttypecode_distribution(self, df):
        incidenttypecode_counts = df['INCIDENTTYPECODE'].value_counts()
        total_count = len(df)
        distribution = {}
        for category, count in incidenttypecode_counts.items():
            percentage = (count / total_count) * 100
            distribution[category] = f"{percentage:.2f}%"
        return distribution

def main():
    predictor = IncidentTypePredictor()

    # Load the data
    df = pd.read_csv('adani_data.csv')

    # Get the distribution of INCIDENTTYPECODE
    distribution = predictor.get_incidenttypecode_distribution(df)

    # Train and export the model
    predictor.train_model(df)

    # Prepare the JSON output
    json_output = {
        "Distribution of INCIDENTTYPECODE": distribution
    }

    # Print the JSON output
    print(json.dumps(json_output, indent=2))

if __name__ == "__main__":
    main()