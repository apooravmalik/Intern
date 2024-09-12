import pandas as pd
from prophet import Prophet

def generate_incident_forecast():
    # Load and prepare data
    data = pd.read_csv('./adani_data.csv')
    data['OCCUREDDATE'] = pd.to_datetime(data['OCCUREDDATE'])
    data = data[data['OCCUREDDATE'].dt.to_period('M') != pd.Period('2024-07')]
    data['month'] = data['OCCUREDDATE'].dt.to_period('M')
    monthly_data = data.groupby('month').size().reset_index(name='incidents')
    monthly_data['month'] = monthly_data['month'].dt.to_timestamp()
    monthly_data.columns = ['ds', 'y']

    # Fit the Prophet model
    model = Prophet()
    model.fit(monthly_data)

    # Create forecast
    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)

    # Prepare response data
    response_data = {
        'historical': monthly_data.to_dict(orient='records'),
        'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6).to_dict(orient='records')
    }
    
    return response_data

# if __name__ == "__main__":
#     forecast_data = generate_incident_forecast()
#     print("Forecast Data:")
#     print(forecast_data)
    