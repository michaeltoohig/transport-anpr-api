import numpy as np
import cv2 as cv
import time


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:           
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids

def detect_objects(net, layer_names, img, show_time: bool=False, confidence_threshold: float=0.5, threshold: float=0.3):
    # Contructing a blob from the input image
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Perform a forward pass of the YOLO object detector
    net.setInput(blob)

    # Getting the outputs from the output layers
    start = time.time()
    outs = net.forward(layer_names)
    end = time.time()

    if show_time:
        print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))
    
    height, width = img.shape[:2]
    # Generate the boxes, confidences, and classIDs
    boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, confidence_threshold)
    
    # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
    idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, threshold)

    detections = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            classId = classids[i]
            confidence = confidences[i]
            detections.append(dict(
                x=x,
                y=y,
                w=w,
                h=h,
                classId=classId,
                confidence=confidence,
            ))
    return detections

# define classes we want to record
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorbike']
OTHER_CLASSES = ['person']

def load_yolo_net():
    labels_file = './data/yolov3/coco-labels'
    config_file = './data/yolov3/yolov3.cfg'
    weights_file = './data/yolov3/yolov3.weights'
    # Get the labels
    labels = open(labels_file).read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv.dnn.readNetFromDarknet(config_file, weights_file)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, labels, colors, layer_names
