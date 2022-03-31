from fastapi import FastAPI
import uvicorn, json
from pydantic import BaseModel
from joblib import load
import pandas as pd

# Start the app
app = FastAPI()

# Specify the input json format as a dict with all the features of the model.
class ListingData(BaseModel):
    host_since :float
    host_response_time : int
    host_response_rate : float
    host_is_superhost : float
    host_listings_count : float
    host_has_profile_pic : float
    host_identity_verified : float
    neighbourhood_cleansed : int
    latitude : float
    longitude : float
    room_type : int
    accommodates : int
    amenities : int
    minimum_nights : int
    maximum_nights : int
    has_availability : int
    availability_30 : int
    availability_60 : int
    availability_90 : int
    availability_365 : int
    number_of_reviews : int
    number_of_reviews_ltm : int
    number_of_reviews_l30d : int
    instant_bookable : int
    reviews_per_month : float
    bathrooms : float
    shared_bath : int


# Specify the input json format as a dict with a key and a value of an array.
class ListingDataArray(BaseModel):
    data: list

# This function runs when you start the server and is responsible for loading the model.
@app.on_event('startup')
def load_model():
    global xgb

    # Load the model.
    xgb = load('models/XGBmodel.joblib')


# This is a health check endpoint that assures that the server is running, for monitoring purposes.
@app.get('/health')
def health():
    return {'status': 'ready'}

# This endpoint just shows the characteristics of the XGB Regression model.
@app.get('/xgb_parameters')
def get_parameters():
    parameters = {
        'colsample_bytree' : xgb.colsample_bytree,
        'learning_rate' : xgb.learning_rate,
        'max_depth' : xgb.max_depth,
        'n_estimators' : xgb.n_estimators
        }   
    
    return parameters

# This endpoint shows the importances of all input features for the XGB Regression model.
@app.get('/xgb_importances')
def get_importances():
    importances = {}
    for n, imp in sorted(zip(xgb.get_booster().feature_names, xgb.feature_importances_), key = lambda x:x[1], reverse=True):
        importances[n] = str(imp)
    return importances

# This is an endpoint that uses the model to make a prediction, but expects the input json in the form
# {
#    'data': [0.34, 0.123, ...]
# }
@app.post('/predict_list')
def predict_list(b: ListingDataArray):
    data = b.data
    res = xgb.predict([data])

    return {"prediction": xgb.target_names[res.item(0)]}

# This is an endpoint that uses the model to make a prediction, but expects the input json in the form
# {
#    'host_since' : 2009.0,
#    'host_response_time' : 2
#    ...
# }
@app.post('/predict')
def predict(sample_data: ListingData):
    # Get data and evaluate it, to turn it into a dictionary.
    response_data = sample_data.json()
    response_data = eval(response_data)

    # Turn the dict into a 1 row DataFrame so it can be used by the model.
    cols = list(response_data.keys())
    data_sample_df = pd.DataFrame(columns = cols)
    data_sample_df.loc[0] = list(response_data.values())

    # Make the prediction.
    res = xgb.predict(data_sample_df)

    # Return the result.
    return {'prediction' : float(res[0])}


if __name__ == '__main__':
    print('Hello')
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
