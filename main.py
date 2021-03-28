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
import json
import time
import base64
import requests
from datetime import datetime
from pathlib import Path
import PySimpleGUI as sg
from PySimpleGUI.PySimpleGUI import ColorChooserButton
from typing import Optional

import typer
import cv2 as cv
import numpy as np
import PIL

from app.yolo_utils2 import VEHICLE_CLASSES, load_yolo_net, detect_objects, draw_detections, crop_detection
from app.wpod_utils import draw_box, load_wpod_net, get_plate

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


def handle_vehicle_selection(window, vehicle_file_path, vehicle_data):
    window['-VEHICLE IMAGE-'].update(data=convert_to_bytes(str(vehicle_file_path), resize=(400, 600)))
    # handle form
    data = vehicle_data.get(str(vehicle_file_path), None)
    if data:
        window['-FORM PLATE-'].update(data['plate'])
        vote = data.get('vote')
        if vote is not None:
            if vote:
                window['-FORM UPVOTE-'].update(True)
            else:
                window['-FORM DOWNVOTE-'].update(True)
        else:
            window['-FORM NOVOTE-'].update(True)
    else:
        window['-FORM PLATE-']('')
        window['-FORM NOVOTE-'](True)
    # handle plate images
    plate_image_pattern = re.compile(f'{vehicle_file_path.name[:3]}p\d\d\.')
    plate_images = [f.name for f in vehicle_file_path.parent.glob('*') if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'] and plate_image_pattern.match(f.name) is not None]
    if plate_images:
        window['-PLATE LIST-'].update(plate_images, set_to_index=0)
        plate_file_path = vehicle_file_path.parent / plate_images[0]
        window['-PLATE IMAGE-'].update(data=convert_to_bytes(str(plate_file_path)))
    else:
        window['-PLATE LIST-']('')
        window['-PLATE IMAGE-']('')


def handle_image_selection(window, directory, vehicle_data):
    vehicle_image_pattern = re.compile('v\d\d\.')
    vehicle_images = [f.name for f in directory.glob('*') if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'] and vehicle_image_pattern.match(f.name) is not None]
    if vehicle_images:
        window['-VEHICLE LIST-'].update(vehicle_images, set_to_index=0)
        vehicle_file_path = directory / vehicle_images[0]
        window['-VEHICLE IMAGE-'].update(data=convert_to_bytes(str(vehicle_file_path), resize=(400, 600)))
        handle_vehicle_selection(window, vehicle_file_path, vehicle_data)
    else:
        window['-VEHICLE LIST-']('')
        window['-VEHICLE IMAGE-']('')
        window['-PLATE LIST-']('')
        window['-PLATE IMAGE-']('')


def open_vehicle_data(directory):
    file = directory + '.json'
    # import pdb; pdb.set_trace()
    if not os.path.exists(file):
        return {}
    with open(file, 'r') as f:
        return json.load(f)


def save_vehicle_data(directory, data):
    file = directory + '.json'
    with open(file, 'w') as f:
        json.dump(data, f)


@cli.command()
def classify(
    folder: Path = typer.Argument("./output")
):
    # --------------------------------- Define Layout ---------------------------------
    left_col = [
        [sg.Text('Folder'), sg.In(size=(15,1), enable_events=True, key='-FOLDER-'), sg.FolderBrowse(initial_folder=str(folder))],
        [sg.Listbox(values=[], enable_events=True, size=(30,10), key='-IMAGE LIST-')],
        [sg.Listbox(values=[], enable_events=True, size=(30,6), key='-VEHICLE LIST-')],
        [sg.Listbox(values=[], enable_events=True, size=(30,6), key='-PLATE LIST-')],
    ]
    images_col = [
        [sg.Image(key='-VEHICLE IMAGE-')],
        [sg.Image(key='-PLATE IMAGE-')],
    ]
    right_col = [
        [sg.Text('Plate'), sg.Input(enable_events=True, key='-FORM PLATE-')],
        [sg.Text('Vote'), sg.Radio('No Vote', 'form_vote', key='-FORM NOVOTE-'), sg.Radio('Upvote', 'form_vote', key='-FORM UPVOTE-'), sg.Radio('Downvote', 'form_vote', key='-FORM DOWNVOTE-')],
        [sg.Button('SAVE')],
    ]
    # ----- Full layout -----
    layout = [
        [
            sg.Column(left_col, element_justification='c',),
            sg.VSeperator(),
            sg.Column(images_col, element_justification='c'),
            sg.VSeperator(),
            sg.Column(right_col),
        ],
    ]
    # --------------------------------- Create Window ---------------------------------
    window = sg.Window('Multiple Format Image Viewer', layout,resizable=True)   
    # --------------------------------- Event Loop ---------------------------------
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == '-FOLDER-':  # Folder name was filled in, make a list of image folders in the folder
            folder = values['-FOLDER-']
            try:
                vehicle_data = open_vehicle_data(folder)
            except:
                print(f'** Error opening vehicle data **')
                break
            try:
                image_dir_list = os.listdir(folder)
            except:
                image_dir_list = []
            window['-IMAGE LIST-'].update(image_dir_list, set_to_index=0)
        elif event == '-IMAGE LIST-':  # An image folder selected; show vehicles within
            try:
                directory = Path(values['-FOLDER-']) / values['-IMAGE LIST-'][0]
                handle_image_selection(window, directory, vehicle_data)
            except Exception as e:
                print(f'**Error {e} **')
                pass
        elif event == '-VEHICLE LIST-':  # A vehicle image selected; show image and plates
            try:
                vehicle_filename = Path(values['-FOLDER-']) / values['-IMAGE LIST-'][0] / values['-VEHICLE LIST-'][0]
                handle_vehicle_selection(window, vehicle_filename, vehicle_data)
            except Exception as e:
                print(f'**Error {e} **')
                pass    
        elif event == 'SAVE':
            vehicle_filename = Path(values['-FOLDER-']) / values['-IMAGE LIST-'][0] / values['-VEHICLE LIST-'][0]
            plate = values['-FORM PLATE-']
            upvote = values['-FORM UPVOTE-']
            downvote = values['-FORM DOWNVOTE-']
            vote = True if upvote else False if downvote else None
            # import pdb; pdb.set_trace()
            vehicle_data[str(vehicle_filename)] = dict(
                plate=plate,
                vote=vote,
            )
            try:
                save_vehicle_data(values['-FOLDER-'], vehicle_data)
            except:
                break
    # --------------------------------- Close & Exit ---------------------------------
    window.close()


def getCropDimensions(h: int, w: int, detection, padding: float = 0.2):
    dx = detection["x"]
    dy = detection["y"]
    dh = detection["h"]
    dw = detection["w"]
    typer.echo(f"Detect x:{dx} y:{dy}, h:{dh} w:{dw}, r:{dw/dh}")

    # Adjust aspect ratio of bounding box
    if (dw / dh) > ASPECT_RATIO:
        fh = dw / ASPECT_RATIO
        fw = dw
    else:
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
    input_dir: Path = typer.Argument(
        ...,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        "./output",
        file_okay=False,
        dir_okay=True,
    ),
    debug: bool = typer.Option(False),
):
    yolo_net, yolo_labels, yolo_colors, yolo_layers = load_yolo_net()
    wpod_net = load_wpod_net()

    if not debug:
        now = datetime.now().timestamp()
        output_path = output_dir / str(int(now))
        output_path.mkdir(parents=True, exist_ok=False)

    for img_path in [f for f in input_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg']]:  # XXX hardcoded filetype
        img = cv.imread(str(img_path))
        h, w = img.shape[:2]
        detections = detect_objects(yolo_net, yolo_labels, yolo_layers, yolo_colors, img)
        detections = list(filter(lambda d: d["label"] in VEHICLE_CLASSES, detections))

        if not debug:
            (output_path / img_path.stem).mkdir()
            cv.imwrite(str(output_path / img_path.stem / f"original{img_path.suffix}" ), img)
        else:
            detections_img = draw_detections(img, detections)
            cv.imshow("Detections", detections_img)

        for num, obj in enumerate(detections):
            typer.echo("*" * 60)
            fx, fy, fh, fw = getCropDimensions(h, w, obj)
            vehicle_img = img[fy:fy+fh, fx:fx+fw].copy()
            if debug:
                cv.imshow("VehicleImage", vehicle_img)
                cv.waitKey(0)
                continue
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


@cli.command()
def upload(
    input_file: Path = typer.Argument(...),
    api_base_url: str = typer.Option("http://localhost:5000", envvar="API_BASE_URL"),
    token: Optional[str] = typer.Option(None, envvar="API_AUTH_TOKEN"),
):
    with open(input_file, "r") as f:
        vehicles = json.load(f)

    if token is None:
        username = typer.prompt('Username')
        password = typer.prompt('Password', hide_input=True)
        url = f"{api_base_url}/v1/auth/login"
        try:
            resp = requests.post(
                url,
                headers=dict(accept="application/json"),
                json=dict(
                    username=username,
                    password=password,
                ),
                timeout=5,
            )
            token = resp.json().get("access_token")
            typer.echo(f"export API_AUTH_TOKEN={token}")
            # os.environ["API_AUTH_TOKEN"] = token
        except:
            raise typer.Abort("Failed to login.")

    headers = dict(
        Authorization=f"Bearer {token}",
        accept="application/json",
    )
    
    for image_path, data  in vehicles.items():
        plate = data["plate"]
        # TODO set 1, 0, -1 on previous step
        vote = data["vote"]
        if vote is not None:
            vote = 1 if vote else -1
        else:
            vote = 0
        try:
            url = f"{api_base_url}/v1/vehicles/{plate}?checkExists=True"
            resp = requests.get(
                url,
                headers=headers,
                timeout=5,
            )
            if resp.status_code == 200:
                vehicle_exists = resp.json().get("data").get("exists")
            else:
                typer.echo(f"Skipping {plate}, unexpected response from API: {resp.json()}")
                break
            typer.echo(f"{plate} exists: {vehicle_exists}")
        except:
            typer.echo("Error fetching vehicle exists.")
            break

        if vehicle_exists:
            try:
                # submit image
                url = f"{api_base_url}/v1/vehicles/{plate}/images"
                image_path = Path(image_path)
                files = {'file': (image_path.name, open(image_path, "rb"))}
                resp = requests.post(
                    url,
                    files=files,
                    headers=headers,
                    timeout=30,
                )
                # submit vote
                url = f"{api_base_url}/v1/vehicles/{plate}/vote"
                resp = requests.post(
                    url,
                    headers=headers,
                    json=dict(vote=vote),
                    timeout=30,
                )
            except:
                break
        else:
            try:
                url = f"{api_base_url}/v1/vehicles/images?detect=False"
                image_path = Path(image_path)
                files = {'file': (image_path.name, open(image_path, "rb"))}
                resp = requests.post(
                    url,
                    files=files,
                    headers=headers,  # {"Authorization": f"Bearer {token}"},
                    timeout=30,
                )
                # import pdb; pdb.set_trace()
                if resp.status_code == 200:
                    url = resp.json().get("data").get("statusUrl")
                    progress = 0
                    while progress < 1:
                        time.sleep(0.5)
                        resp = requests.get(
                            f"{api_base_url}{url}",
                            headers=headers,
                            timeout=5,
                        )
                        progress = float(resp.json().get("data").get("progress"))
                    filename = resp.json().get("data").get("filename")
                    url = f"{api_base_url}/v1/vehicles"
                    resp = requests.post(
                        url,
                        headers=headers,
                        json=dict(
                            plate=plate,
                            image=dict(
                                filename=filename,
                            ),
                            # comment=dict(body=comment),  # not yet collected
                            vote=dict(vote=vote),
                        ),
                        timeout=5,
                    )
                else:
                    print(resp.json())
                    typer.echo("Uploading image error")
                    break
            except:
                print(resp.json())
                typer.echo("Error uploading vehicle image.")
                break
        # image_path = Path(image_path)
        # url = f"{API_BASE_URL}/v1/api/vehicles/images"
        # files = {'image': (image_path.name, open(image_path, "rb"))}
        # try:
        #     resp = requests.post(url, files=files, timeout=30)
        #     if resp.status_code == 201:
        #         data = resp.json()
        #         anprStatusUrl = data.get("statusUrl", "")
        # except:
        #     pass
    
    
if __name__ == "__main__":
    cli()
