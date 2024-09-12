import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IncidentCategoryPredictor:
    def __init__(self):
        self.rf_model = None
        self.le = None
        self.feature_columns = None

    def load_model(self):
        try:
            self.rf_model = joblib.load('rf_model_incident_category.joblib')
            self.le = joblib.load('label_encoder_incident_category.joblib')
            self.feature_columns = joblib.load('feature_columns_incident_category.joblib')
            logging.info("Model loaded successfully")
        except FileNotFoundError:
            logging.error("Model files not found. Please train the model first.")
            raise

    def train_model(self, df):
        logging.info("Starting model training")
        # Data Preparation and Feature Engineering
        date_columns = ['REPORTEDDATE', 'LASTUPDATEDDATE', 'OCCUREDDATE']
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')

        df['Day_of_Week'] = df['OCCUREDDATE'].dt.dayofweek
        df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)

        categorical_cols = ['PRIORITY', 'BVCODE', 'BUNAME', 'SICODE', 'LMCODE', 'INCIDENTTYPECODE']
        df_encoded = pd.get_dummies(df, columns=categorical_cols)

        self.le = LabelEncoder()
        df_encoded['INCIDENTCATCODE'] = self.le.fit_transform(df['INCIDENTCATCODE'])

        X = df_encoded.drop(['INCIDENTCATCODE', 'REPORTEDDATE', 'LASTUPDATEDDATE', 'OCCUREDDATE'], axis=1)
        y = df_encoded['INCIDENTCATCODE']

        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)

        self.feature_columns = list(X.columns)

        # Save the model, label encoder, and feature columns
        joblib.dump(self.rf_model, 'rf_model_incident_category.joblib')
        joblib.dump(self.le, 'label_encoder_incident_category.joblib')
        joblib.dump(self.feature_columns, 'feature_columns_incident_category.joblib')
        logging.info("Model training completed and saved")

    def predict_future_incidents(self, start_date, end_date):
        if self.rf_model is None or self.le is None or self.feature_columns is None:
            self.load_model()

        future_dates = pd.date_range(start=start_date, end=end_date)
        future_df = pd.DataFrame({'OCCUREDDATE': future_dates})
        future_df['Day_of_Week'] = future_df['OCCUREDDATE'].dt.dayofweek
        future_df['Is_Weekend'] = future_df['Day_of_Week'].isin([5, 6]).astype(int)

        categorical_cols = ['PRIORITY', 'BVCODE', 'BUNAME', 'SICODE', 'LMCODE', 'INCIDENTTYPECODE']
        for col in categorical_cols:
            future_df[col] = np.random.choice(['A', 'B', 'C'], size=len(future_df))  # Replace with actual categories

        future_df_encoded = pd.get_dummies(future_df, columns=categorical_cols)

        for col in self.feature_columns:
            if col not in future_df_encoded.columns:
                future_df_encoded[col] = 0

        future_df_encoded = future_df_encoded[self.feature_columns]

        future_predictions = self.rf_model.predict_proba(future_df_encoded)
        future_df['Predicted_Category'] = self.le.inverse_transform([np.random.choice(len(self.rf_model.classes_), p=probs) for probs in future_predictions])

        monthly_predictions = future_df.groupby([future_df['OCCUREDDATE'].dt.to_period('M'), 'Predicted_Category']).size().unstack(fill_value=0)

        return monthly_predictions

    def get_predictions(self, start_date, end_date):
        logging.info(f"Generating predictions from {start_date} to {end_date}")
        try:
            monthly_predictions = self.predict_future_incidents(start_date, end_date)

            results = {
                "monthly_predictions": monthly_predictions.to_dict(),
                "monthly_total_predictions": monthly_predictions.sum(axis=1).to_dict(),
                "total_predictions_per_category": monthly_predictions.sum().to_dict()
            }
            logging.info("Predictions generated successfully")
            return results
        except Exception as e:
            logging.error(f"Error generating predictions: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    predictor = IncidentCategoryPredictor()
    
    # If you need to train the model:
    df = pd.read_csv('adani_data.csv')
    predictor.train_model(df)
    
    # To make predictions:
    start_date = datetime.now()
    end_date = start_date + timedelta(days=180)
    
    try:
        predictions = predictor.get_predictions(start_date, end_date)
        
        print("Monthly predictions by category:")
        print(predictions["monthly_predictions"])

        print("\nTotal monthly predictions:")
        print(predictions["monthly_total_predictions"])

        print("\nTotal predictions per category:")
        print(predictions["total_predictions_per_category"])
    except Exception as e:
        print(f"An error occurred: {str(e)}")