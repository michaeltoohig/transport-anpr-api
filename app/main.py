from typing import Optional

import os
import cv2
import numpy as np

from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel

from app.local_utils import detect_plate


def load_model(path):
    try:
        path = os.path.splitext(path)[0]
        with open(f"{path}.json", 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)


wpod_net_path = "data/wpod-net.json"
wpod_net = load_model(wpod_net_path)

# Load model architecture, weight and labels
json_file = open('data/MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("data/License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('data/license_character_classes.npy')
print("[INFO] Labels loaded successfully...")


def preprocess_image(img, resize=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.bitwise_not(img)  # TODO determine if bitwise_not is required if plate is white on black or reverse
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


def get_plate(img, Dmax=608, Dmin = 608):
    vehicle_img = preprocess_image(img)
    ratio = float(max(vehicle_img.shape[:2])) / min(vehicle_img.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , plate_img, _, cor = detect_plate(wpod_net, vehicle_img, bound_dim, lp_threshold=0.5)
    return vehicle_img, plate_img, cor


# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts


# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction


app = FastAPI()


class PlatePrediction(BaseModel):
    predition: str


@app.post("/plate")  #, response_model=PlatePrediction)
def read_root(
    image: UploadFile = File(...), 
    token: str = Form(...)
):
    img = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    try:
        vehicle_img, plate_img, cor = get_plate(img)
    except AssertionError:
        raise HTTPException(
            status_code=404,
            detail="Number plate not found in vehicle image"
        )
    if not len(plate_img): # Check if there is at least one license image
        raise HTTPException(
            status_code=404,
            detail="Number plate not found in vehicle image"
        )

    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_image = cv2.convertScaleAbs(plate_img[0], alpha=(255.0))
    
    # Convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    
    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()

    # Initialize a list which will be used to append charater image
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 30, 60

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=3.5: # Only select contour with defined ratio
            if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                # Sperate number and gibe prediction
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    print("Detect {} letters...".format(len(crop_characters)))

    final_string = ""
    for i,character in enumerate(crop_characters):
        title = np.array2string(predict_from_model(character ,model ,labels))
        final_string += title.strip("'[]")
    
    return {"prediction": final_string}