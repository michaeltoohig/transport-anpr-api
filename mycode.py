import cv2 as cv
from app.yolo_utils2 import load_yolo_net, detect_objects
from app.wpod_utils import get_plate 
from app.other_utils import draw_box

IMAGE_PATH = "./bus1.jpg"
net, labels, colors, layer_names = load_yolo_net()

img = cv.imread(IMAGE_PATH)
cv.imshow("Original", img)
#cv.waitKey(1000)

objs = detect_objects(net, layer_names, img)

for obj in objs:
    x = max(obj["x"], 0)
    y = max(obj["y"], 0)  # if bounding box extends out of image then start at 0
    h = obj["h"]
    w = obj["w"]
    vehicle_img = img[y:y+h, x:x+w].copy()
    cv.imshow("VehicleImage", vehicle_img)
    #cv.waitKey(1000)
    plate_images, cor = get_plate(vehicle_img)
    img = draw_box(vehicle_img, cor)
    cv.imshow(" ", img)
    for num, plate_img in enumerate(plate_images):
        cv.imshow(f"Plate: {num}", plate_img)
    cv.waitKey(0)