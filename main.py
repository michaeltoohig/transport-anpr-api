"""
Run the app locally via CLI.

For batch processing directory of images or for
handling video files.
"""

# import app.cli.main as commands

# if __name__ == "__main__":
#     commands.cli()

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import cv2 as cv
import numpy as np

from app.yolo_utils2 import VEHICLE_CLASSES, load_yolo_net, detect_objects, draw_detections, crop_detection
from app.wpod_utils import load_wpod_net, get_plate

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
    wpod_net = load_wpod_net()

    now = datetime.now().timestamp()
    output_path = output_dir / str(int(now))
    output_path.mkdir(parents=True, exist_ok=False)

    for img_path in input_dir.glob('*.jpg'):  # XXX hardcoded filetype
        img = cv.imread(str(img_path))
        (output_path / img_path.stem).mkdir()
        cv.imwrite(str(output_path / img_path.stem / f"original{img_path.suffix}" ), img)
        h, w = img.shape[:2]
        detections = detect_objects(yolo_net, yolo_labels, yolo_layers, yolo_colors, img)
        detections = list(filter(lambda d: d["label"] in VEHICLE_CLASSES, detections))

        for num, obj in enumerate(detections):
            typer.echo("*" * 60)
            dx = obj["x"]
            dy = obj["y"]
            dh = obj["h"]
            dw = obj["w"]

            # Adjust aspect ratio of bounding box
            fh = dh
            fw = dh * ASPECT_RATIO 

            # Calculate bounding box padding
            paddingRatio = 0.3  # XXX hardcoded
            fh += fh * paddingRatio
            fw += fw * paddingRatio

            # Center new larger bounding box over old one
            fy = dy + ((dh - fh) / 2)
            fx = int(dx + ((dw - fw) / 2))

            # Shift bounding area if it exceeds image frame
            if fx < 0:
                if abs(fx) + fw > w:
                    # width is greater than total width - adjust height and width
                    fw = w
                    fx = 0
                    newHeight = fw / ASPECT_RATIO
                    fy += ((fh - newHeight) / 2)
                    fh = newHeight
                else:
                    # shift right
                    fx = 0

            if fy < 0:
                if abs(fy) + fh > h:
                    # height is greater than total height - adjust height and width
                    fh = h
                    fy = 0
                    newWidth = fh * ASPECT_RATIO
                    fx += ((fw - newWidth) / 2)
                    fw = newWidth
                else:
                    # shift down
                    fy = 0  

            fx, fy, fh, fw = int(fx), int(fy), int(fh), int(fw)
            typer.echo(f"Detect x:{dx} y:{dy}, h:{dh} w:{dw}, r:{dw/dh}")
            typer.echo(f"Final  x:{fx} y:{fy}, h:{fh} w:{fw}, r:{fw/fh}")
            vehicle_img = img[fy:fy+fh, fx:fx+fw].copy()
            # cv.imshow("VehicleImage", vehicle_img)
            vehicle_img_path = output_path / img_path.stem / f"v{num:02}{img_path.suffix}"
            cv.imwrite(str(vehicle_img_path), vehicle_img)
            try:
                plate_images, _ = get_plate(wpod_net, vehicle_img)
                for num2, plate_img in enumerate(plate_images):
                    print(f"plate {num2:02}")
                    img_float32 = np.float32(plate_img)
                    plate_img = cv.cvtColor(img_float32, cv.COLOR_RGB2BGR)
                    # cv.imshow(f"Plate: {num}", plate_img)
                    plate_img_path = output_path / img_path.stem / f"v{num:02}p{num2:02}{img_path.suffix}"
                    cv.imwrite(str(plate_img_path), plate_img * 255)
            except:
                typer.echo('No plate found')
            # cv.waitKey(0)


if __name__ == "__main__":
    cli()
