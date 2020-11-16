import cv2
import numpy as np
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder


# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts


# pre-processing input images and pedict with model
def predict_from_model(image, model, labels):
    # Resize image to size expected for OCR
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction


def get_prediction(plate_img):
    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_image = cv2.convertScaleAbs(plate_imgd, alpha=(255.0))

    # Convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)

    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # creat a copy version "test_roi" of plate_image to draw bounding box
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

                # Sperate numb-er and gibe prediction
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    print("Detect {} letters...".format(len(crop_characters)))

    final_string = ""
    for i, character in enumerate(crop_characters):
        print(character)
        title = np.array2string(predict_from_model(character, model, labels))
        final_string += title.strip("'[]")
    return final_string


def load_ocr_net():
    mobilenets_ocr_path = "data/MobileNets_character_recognition.json"
    mobilenets_ocr_weights_path = "data/License_character_recognition_weight.h5"
    ocr_classes_path = "data/license_character_classes.npy"
    
    ocr_net = model_from_json(open(mobilenets_ocr_path, "r").read())
    ocr_net.load_weights(mobilenets_ocr_weights_path)
    print("[INFO] mobilenet_ocr loaded successfully...")

    labels = LabelEncoder()
    labels.classes_ = np.load(ocr_classes_path)
    print("[INFO] labels loaded successfully...")
    return ocr_net, labels