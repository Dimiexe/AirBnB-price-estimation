# Import the web framework
from fastapi import FastAPI
# Import the server
import uvicorn
# This is a library that does data vlidation
from pydantic import BaseModel

from joblib import load
import pandas as pd

# Start the app
app = FastAPI()

# Specify the input json format as a dict with all the features of the model
class BreastData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

# Specify the input json format as an dict with a key and a value of an array
class BreastDataArray(BaseModel):
    data: list

# This function runs when you start the server and is responsible for loading the model
@app.on_event('startup')
def load_model():
    global rf
    rf = load('data/rf1.joblib')

# The following @app.get('/blah') or app.post('/blah')
# specify endpoints or url paths that can be accessed with GET or POST requests
# Usuall GET is getting information for the server, whereas POST sends some data in the server
# the server does some processing, or saving of that data and returns something

# This is a health check endpoint that assures that the server is running. Usually for monitoring purposes
@app.get('/health')
def health():
    return {'status': 'ready'}

# This endpoint just shows the importances of the features of the RandomForest Classifier.
# This can be used to be sure that we have the correct model loaded, as well as provide
# a visualization/justification of the important features for our model. Not all ML models
# have this attribute
@app.get('/rf_importances')
def get_importances():
    print('hi')
    features_importances = {}
    for n, imp in sorted(zip(rf.feature_names, rf.feature_importances_), key= lambda x: x[1], reverse=True):
        features_importances[n] = imp
    
    return features_importances

# This is an endpoint that uses the model to make a prediction, but expects the input json in the form
# {
#    'data': [0.34, 0.123, ...]
# }
@app.post('/predict_list')
def predict_list(b: BreastDataArray):
    data = b.data
    res = rf.predict([data])

    return {"prediction": rf.target_names[res.item(0)]}

# This is an endpoint that uses the model to make a prediction, but expects the input json in the form
# {
#    'mean_radious': 0.34,
#    'worst_area': 0.21,
#    ...
# }
@app.post('/predict')
def predict(b: BreastData):
    new_case = b.json()
    new_case = eval(new_case)
  
    for k, v in new_case.items():
        assert type(v) == float
  
    data = pd.Series(new_case).values
 
    res = rf.predict([data])

    
    return {"prediction": rf.target_names[res.item(0)]}


if __name__ == '__main__':
    print('yo')
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
