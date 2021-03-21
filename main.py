"""
Run the app locally via CLI.

For batch processing directory of images or for
handling video files.
"""

# import app.cli.main as commands

# if __name__ == "__main__":
#     commands.cli()

from pathlib import Path
from typing import Optional

import typer
import cv2 as cv

from app.yolo_utils2 import VEHICLE_CLASSES, load_yolo_net, detect_objects, draw_detections, crop_detection
from app.wpod_utils import load_wpod_net, get_plate, draw_box

cli = typer.Typer()

ASPECT_RATIO = 4 / 3


def getCropDimensions(h: int, w: int, detection, padding: float = 0.2):
    dx = detection["x"]
    dy = detection["y"]
    dw = detection["w"]
    dh = detection["h"]
    # padding required
    padw = padding * dw
    padh = padding * dh 

    max_width = max(w, (dw + (padw * 2)))
    max_height = max(h, (dh + (padh * 2)))

    # crop dimensions
    cx = dx - padw
    cw = dw + padw
    
    adjust_right = None
    if cx < 0:
        # crop beyond left border of image
        adjust_right = abs(cx)
        cx = 0
    adjust_left = None
    if dx + cw > w:
        # crop beyond right border of image
        adjust_left = abs((dx + cw) - w)
        cw = w

    if adjust_right and adjust_left is None:
        if adjust_right + cw < w:
            cw = cw + adjust_right
    
    if adjust_left and adjust_left is None:
        if cx - adjust_left > 0:
            cw = cw - adjust_left

    
    cy = dy - padh
    ch = dh + padh
    
    if ((cw / ch) > ASPECT_RATIO):
        # more wide than tall - add to height
        pass


@cli.command()
def images(
    input_dir: Path = typer.Option(
        ...,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        file_okay=False,
        dir_okay=True,
    ),
):
    yolo_net, yolo_labels, yolo_colors, yolo_layers = load_yolo_net()

    for img_path in input_dir.glob('*.jpg'):  # XXX hardcoded filetype
        print(img_path)
        img = cv.imread(str(img_path))
        h, w = img.shape[:2]
        detections = detect_objects(yolo_net, yolo_labels, yolo_layers, yolo_colors, img)
        detections = list(filter(lambda d: d["label"] in VEHICLE_CLASSES, detections))

        for num, obj in enumerate(detections):
            # TODO crop with buffer area around detection in 4/3 aspect ratio
            print(obj)
            dx = max(obj["x"], 0)
            dy = max(obj["y"], 0)
            dh = obj["h"]
            dw = obj["w"]

            aspect = w / h
            if aspect > ASPECT_RATIO:
                padding = round(dh * 0.2)
            else:
                padding = round(dw * 0.2)

            dx = dx - padding
            dy = dy - padding
            dh = dh + (padding * 2)
            dw = dw + (padding * 2)
            
            # if aspect > ASPECT_RATIO:
            #     # reduce width or increase height
            #     h = h + padding
            # else:
            #     # reduce height or increase width
            #     w = w + padding

            vehicle_img = img[dy:dy+dh, dx:dx+dw].copy()
            cv.imshow("VehicleImage", vehicle_img)
            cv.waitKey(0)


if __name__ == "__main__":
    cli()
