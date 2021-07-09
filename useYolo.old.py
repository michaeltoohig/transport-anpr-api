"""
Original script I modified for my use. Supports single image, video, or webcam views. Not really anything I would use now.
"""

import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os

from app.yolo_utils import detect_objects, draw_detections, load_yolo_net

FLAGS = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-path',
        type=str,
        default='./data/yolov3/',
        help='The directory where the model weights and \
              configuration files are.')

    parser.add_argument('-w', '--weights',
        type=str,
        default='./data/yolov3/yolov3.weights',
        help='Path to the file which contains the weights \
                 for YOLOv3.')

    parser.add_argument('-cfg', '--config',
        type=str,
        default='./data/yolov3/yolov3.cfg',
        help='Path to the configuration file for the YOLOv3 model.')

    parser.add_argument('-i', '--image-path',
        type=str,
        help='The path to the image file')

    parser.add_argument('-v', '--video-path',
        type=str,
        help='The path to the video file')


    parser.add_argument('-vo', '--video-output-path',
        type=str,
        default='./output.avi',
        help='The path of the output video file')

    parser.add_argument('-l', '--labels',
        type=str,
        default='./data/yolov3/coco-labels',
        help='Path to the file having the \
                    labels in a new-line seperated way.')

    parser.add_argument('-c', '--confidence',
        type=float,
        default=0.5,
        help='The model will reject boundaries which has a \
                probabiity less than the confidence value. \
                default: 0.5')

    parser.add_argument('-th', '--threshold',
        type=float,
        default=0.3,
        help='The threshold to use when applying the \
                Non-Max Suppresion')

    parser.add_argument('--download-model',
        type=bool,
        default=False,
        help='Set to True, if the model weights and configurations \
                are not present on your local machine.')

    parser.add_argument('-t', '--show-time',
        type=bool,
        default=False,
        help='Show the time taken to infer each image.')

    FLAGS, unparsed = parser.parse_known_args()

    yolo_net, yolo_labels, yolo_colors, yolo_layers = load_yolo_net()
    # net, labels, colors, layer_names = get_yolo_net(FLAGS.labels, FLAGS.config, FLAGS.weights)
        
    # If both image and video files are given then raise error
    if FLAGS.image_path is None and FLAGS.video_path is None:
        print ('Neither path to an image or path to video provided')
        print ('Starting Inference on Webcam')

    # Do inference with given image
    if FLAGS.image_path:
        # Read the image
        try:
            img = cv.imread(FLAGS.image_path)
            height, width = img.shape[:2]
        except:
            raise 'Image cannot be loaded!\n\
                               Please check the path provided!'

        finally:
            img, _, _, _, _, _ = infer_image(net, layer_names, height, width, img, colors, labels, FLAGS.show_time, FLAGS.confidence, FLAGS.threshold)
            show_image(img)

    elif FLAGS.video_path:
        # Read the video
        try:
            vid = cv.VideoCapture(FLAGS.video_path)
            height, width = None, None
            writer = None
        except:
            raise 'Video cannot be loaded!\n\
                               Please check the path provided!'

        finally:
            count = 0
            while True:
                grabbed, frame = vid.read()
                # Checking if the complete video is read
                if not grabbed:
                    break

                if width is None or height is None:
                    height, width = frame.shape[:2]

                if count == 0:
                    print('detect')
                    # frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS.show_time, FLAGS.confidence, FLAGS.threshold)
                    detections = detect_objects(yolo_net, yolo_labels, yolo_layers, yolo_colors, frame, FLAGS.show_time, FLAGS.confidence, FLAGS.threshold)
                    draw_detections(frame, detections)
                    count += 1
                    
                    # Below is now in the `draw_labels_and_boxes` function
                    # if len(idxs) > 0:  # same as draw labels_and_boxes so could maybe keep code there instead of separated like this
                    #     for i in idxs.flatten():
                    #         x, y = boxes[i][0], boxes[i][1]
                    #         w, h = boxes[i][2], boxes[i][3]
                    #         vehicle_box = frame[y:y+h, x:x+w]
                    #         try:
                    #             predictions = get_prediction(vehicle_box)
                    #             print(predictions)
                    #         except AssertionError:
                    #             pass
                else:
                    # frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS.show_time, FLAGS.confidence, FLAGS.threshold, boxes, confidences, classids, idxs, infer=False)
                    # detections = detect_objects(yolo_net, yolo_labels, yolo_layers, yolo_colors, frame, FLAGS.show_time, FLAGS.confidence, FLAGS.threshold)
                    draw_detections(frame, detections)
                    count = (count + 1) % 6

                    
                if writer is None:
                    # Initialize the video writer
                    fourcc = cv.VideoWriter_fourcc(*"MJPG")
                    writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, 
                                    (frame.shape[1], frame.shape[0]), True)

                writer.write(frame)

            print ("[INFO] Cleaning up...")
            writer.release()
            vid.release()


    else:
        # Infer real-time on webcam
        count = 0

        vid = cv.VideoCapture(0)
        while True:
            _, frame = vid.read()
            height, width = frame.shape[:2]

            if count == 0:
                frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
                                    height, width, frame, colors, labels, FLAGS.show_time, FLAGS.confidence, FLAGS.threshold)
                count += 1
            else:
                frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
                                    height, width, frame, colors, labels, FLAGS.show_time, FLAGS.confidence, FLAGS.threshold, boxes, confidences, classids, idxs, predictions, infer=False)
                count = (count + 1) % 6

            cv.imshow('webcam', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv.destroyAllWindows()