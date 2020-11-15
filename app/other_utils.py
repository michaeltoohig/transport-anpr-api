import cv2
import numpy as np

# in wpod_utils now
def preprocess_image(img, resize=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.bitwise_not(img)  # TODO determine if bitwise_not is required if plate is white on black or reverse
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def draw_box(image_path, cor, thickness=3): 
    pts=[]  
    x_coordinates=cor[0][0]
    y_coordinates=cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
    
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    vehicle_image = preprocess_image(image_path)
    
    cv2.polylines(vehicle_image,[pts],True,(0,255,0),thickness)
    return vehicle_image

# In yolo utils; not needed here or is this reuseable ?
def draw_label_and_box(img, x, y, w, h, label: str):
    """
    This should be agnostic to the NN that asks for the box to be drawn... but the old function was heavily tied to the implementation.
    """
    # If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img