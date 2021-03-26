"""
Run the app locally via CLI.

For batch processing directory of images or for
handling video files.
"""

# import app.cli.main as commands

# if __name__ == "__main__":
#     commands.cli()

import io
import re
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional
import PySimpleGUI as sg

import typer
import cv2 as cv
import numpy as np
import PIL

from app.yolo_utils2 import VEHICLE_CLASSES, load_yolo_net, detect_objects, draw_detections, crop_detection
from app.wpod_utils import load_wpod_net, get_plate

cli = typer.Typer()

ASPECT_RATIO = 4 / 3


def convert_to_bytes(file_or_bytes, resize=None):
    """
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    """
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()


@cli.command()
def classify():
    # --------------------------------- Define Layout ---------------------------------
    # First the window layout...2 columns

    left_col = [
        [sg.Text('Folder'), sg.In(size=(25,1), enable_events=True, key='-FOLDER-'), sg.FolderBrowse()],
        [sg.Listbox(values=[], enable_events=True, size=(40,10), key='-IMAGE LIST-')],
        [sg.Listbox(values=[], enable_events=True, size=(40,10), key='-VEHICLE LIST-')],
    ]

    # For now will only show the name of the file that was chosen
    images_col = [[sg.Text('You choose from the list:')],
                [sg.Text(size=(40,1), key='-TOUT-')],
                [sg.Image(key='-IMAGE-')]]

    # ----- Full layout -----
    layout = [[sg.Column(left_col, element_justification='c'), sg.VSeperator(),sg.Column(images_col, element_justification='c')]]

    # --------------------------------- Create Window ---------------------------------
    window = sg.Window('Multiple Format Image Viewer', layout,resizable=True)

    
    # ----- Run the Event Loop -----
    vehicle_image_pattern = re.compile('v\d\d\.')
    plate_image_pattern = re.compile('v\d\dp\d\d\.')
    # --------------------------------- Event Loop ---------------------------------
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == '-FOLDER-':                         # Folder name was filled in, make a list of image folders in the folder
            folder = values['-FOLDER-']
            try:
                image_dir_list = os.listdir(folder)
            except:
                image_dir_list = []
            # fnames = [f for f in file_list if os.path.isfile(
            #     os.path.join(folder, f)) and f.lower().endswith((".png", ".jpg", "jpeg", ".tiff", ".bmp"))]
            window['-IMAGE LIST-'].update(image_dir_list)
        elif event == '-IMAGE LIST-':  # An image folder selected; show vehicles within
            try:
                # import pdb; pdb.set_trace()
                image_dir = Path(values['-FOLDER-']) / values['-IMAGE LIST-'][0]
                vehicle_images = [f.name for f in image_dir.glob('*') if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'] and vehicle_image_pattern.match(f.name) is not None]
                # import pdb; pdb.set_trace()
                # window.['-TOUT-'].update(vehicle_images[0])
                window['-VEHICLE LIST-'].update(vehicle_images)
                # window['-IMAGE-'].update(data=convert_to_bytes(str(vehicle_images[0]), resize=(200, 200)))
            except Exception as e:
                print(f'**Error {e} **')
                pass
        elif event == '-VEHICLE LIST-':  # A vehicle image selected; show image and plates
            try:
                filename = Path(values['-FOLDER-']) / values['-IMAGE LIST-'][0] / values['-VEHICLE LIST-'][0]
                window['-IMAGE-'].update(data=convert_to_bytes(str(filename), resize=(600, 600)))
            except Exception as e:
                print(f'**Error {e} **')
                pass    

        # elif event == '-FILE LIST-':    # A file was chosen from the listbox
        #     try:
        #         filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
        #         window['-TOUT-'].update(filename)
        #         if values['-W-'] and values['-H-']:
        #             new_size = int(values['-W-']), int(values['-H-'])
        #         else:
        #             new_size = None
        #         window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=new_size))
        #     except Exception as E:
        #         print(f'** Error {E} **')
        #         pass        # something weird happened making the full filename
    # --------------------------------- Close & Exit ---------------------------------
    window.close()


def getCropDimensions(h: int, w: int, detection, padding: float = 0.2):
    dx = detection["x"]
    dy = detection["y"]
    dh = detection["h"]
    dw = detection["w"]
    typer.echo(f"Detect x:{dx} y:{dy}, h:{dh} w:{dw}, r:{dw/dh}")

    # Adjust aspect ratio of bounding box
    fh = dh
    fw = dh * ASPECT_RATIO 

    # Calculate bounding box padding
    fh += fh * padding
    fw += fw * padding

    # Center new larger bounding box over old one
    fy = dy + ((dh - fh) / 2)
    fx = int(dx + ((dw - fw) / 2))

    # Shift bounding area if it exceeds image frame
    if fx < 0:
        if abs(fx) + fw > w:
            # width is greater than frame width - adjust height and width
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
            # height is greater than frame height - adjust height and width
            fh = h
            fy = 0
            newWidth = fh * ASPECT_RATIO
            fx += ((fw - newWidth) / 2)
            fw = newWidth
        else:
            # shift down
            fy = 0

    # TODO shift up and shift left?

    fx, fy, fh, fw = int(fx), int(fy), int(fh), int(fw)
    typer.echo(f"Final  x:{fx} y:{fy}, h:{fh} w:{fw}, r:{fw/fh}")
    return fx, fy, fh, fw


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
            fx, fy, fh, fw = getCropDimensions(h, w, obj)
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
