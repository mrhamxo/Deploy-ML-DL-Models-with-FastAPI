from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Annotated
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# load the models
ml_model = joblib.load('models/ml_model.pkl')
dl_model = load_model('models/dl_model.h5')

# initialize FASTAPI app
app = FastAPI()

# enable cors middleware (allow frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods
    allow_headers=["*"],  # allow all headers
)

# class name for iris dataset
class_names = ['setosa', 'versicolor', 'virginica']

# request body for iris dataset
class IrisRequest(BaseModel):
    sepal_length: Annotated[float, Field(..., gt=0, description="Length of the sepal in cm")]
    sepal_width: Annotated[float, Field(..., gt=0, description="Width of the sepal in cm")]
    petal_length: Annotated[float, Field(..., gt=0, description="Length of the petal in cm")]
    petal_width: Annotated[float, Field(..., gt=0, description="Width of the sepal in cm")]
    
@app.post("/predict_iris")
def predict_iris(request: IrisRequest):
    data = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])
    
    prediction = ml_model.predict(data)[0]
    
    # Return prediction as JSON response
    return JSONResponse(status_code=200, content={'predicted_category': class_names[prediction]})

@app.post("/predict_digit")
async def predict_digit(file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()
    
    # Convert bytes to numpy array
    np_arr = np.frombuffer(contents, np.uint8)
    
    # Decode the image
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    
    # Resize and normalize the image
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28)
    
    # predict
    prediction = np.argmax(dl_model.predict(img))
    
    # Return prediction as JSON response
    return JSONResponse(status_code=200, content={'predicted_category': int(prediction)})
