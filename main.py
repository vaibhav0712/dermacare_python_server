# Importing the required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import sys
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
import os
from fastapi.responses import JSONResponse

# FastAPI instance
app = FastAPI()

# Global Configurations
origins = [
    "http://localhost:5173",
]
sys.stdout.reconfigure(encoding='utf-8')
categories = ['BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', "VI-shingles"]
UPLOAD_FOLDER = "upload"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Adding CORS middleware to the FastAPI instance
# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = tf.keras.models.load_model('mobv2.keras')


# Api Routes
@app.get("/")
async def root():
    return {"message": "Hello this is the root of the API"}


def get_confidence(preds):
    top_class = preds.argsort()[0][-1:]
    confidence = preds[0, top_class[0]]
    return confidence*100

@app.post('/predict')
async def predict(file: UploadFile = File(...)):

    # Save the image to the upload folder
    contents = await file.read()
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filename, "wb") as f:
        f.write(contents)

    # Create the image path
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Load the image and preprocess it
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict the image
    prediction = model.predict(x)
    predicted_classes = np.argmax(prediction, axis=1)
    predicted_category = categories[predicted_classes[0]]
    confidence = get_confidence(prediction)
    print(predicted_category,str(confidence))
    os.remove(filename)
    return JSONResponse(content={"message": "success", "prediction": predicted_category, "confidence": str(confidence)})

if __name__ == "__main__":
    import uvicorn
    print("got here")
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=3000)
