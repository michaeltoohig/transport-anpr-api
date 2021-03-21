import argparse

import norfair
import numpy as np
from norfair import Detection, Tracker, Video

from app.yolo_utils2 import detect_objects, draw_detections, load_yolo_net


MAX_DISTANCE_BETWEEN_POINTS = 60


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def get_centroid(yolo_box, img_height, img_width):
    # x1 = yolo_box[0] * img_width
    # y1 = yolo_box[1] * img_height
    # x2 = yolo_box[2] * img_width
    # y2 = yolo_box[3] * img_height
    x1 = yolo_box['x']
    y1 = yolo_box['y']
    x2 = yolo_box['x'] + yolo_box['w']
    y2 = yolo_box['y'] + yolo_box['h']
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


FLAGS = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video-path',
        type=str,
        help='The path to the video file',
    )


    parser.add_argument('-vo', '--video-output-path',
        type=str,
        default='./output.avi',
        help='The path of the output video file',
    )

    parser.add_argument('-i', '--images-path',
        type=str,
        help='The path to the images directory',
    )

    parser.add_argument('-io', '--images-output-path',
        type=str,
        help='The path of the output image files',
    )

    parser.add_argument('-c', '--confidence',
        type=float,
        default=0.5,
        help='The model will reject boundaries which has a \
                probabiity less than the confidence value. \
                default: 0.5',
    )

    parser.add_argument('-th', '--threshold',
        type=float,
        default=0.3,
        help='The threshold to use when applying the \
                Non-Max Suppresion',
    )

    parser.add_argument('-t', '--show-time',
        type=bool,
        default=False,
        help='Show the time taken to infer each image.',
    )

    FLAGS, unparsed = parser.parse_known_args()

    yolo_net, yolo_labels, yolo_colors, yolo_layers = load_yolo_net()

    if FLAGS.video_path:
        video = Video(input_path=FLAGS.video_path, output_path=FLAGS.video_output_path)
        tracker = Tracker(
            distance_function=euclidean_distance,
            distance_threshold=MAX_DISTANCE_BETWEEN_POINTS,
        )

        for frame in video:
            detections = detect_objects(yolo_net, yolo_labels, yolo_layers, yolo_colors, frame, FLAGS.show_time, FLAGS.confidence, FLAGS.threshold)
            detections = [
                Detection(get_centroid(box, frame.shape[0], frame.shape[1]), data=box)
                for box in detections
                # if box[-1] == 2
            ]
            tracked_objects = tracker.update(detections=detections)
            norfair.draw_points(frame, detections)
            norfair.draw_tracked_objects(frame, tracked_objects)
            video.write(frame)
        
    elif FLAGS.images_path:
        print(FLAGS.images_path)
        # finally:
        #     count = 0
        #     while True:
        #         grabbed, frame = vid.read()
        #         # Checking if the complete video is read
        #         if not grabbed:
        #             break

        #         if width is None or height is None:
        #             height, width = frame.shape[:2]

        #         if count == 0:
        #             print('detect')
        #             # frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS.show_time, FLAGS.confidence, FLAGS.threshold)
        #             detections = detect_objects(yolo_net, yolo_labels, yolo_layers, yolo_colors, frame, FLAGS.show_time, FLAGS.confidence, FLAGS.threshold)
        #             draw_detections(frame, detections)
        #             count += 1
                    
        #             # Below is now in the `draw_labels_and_boxes` function
        #             # if len(idxs) > 0:  # same as draw labels_and_boxes so could maybe keep code there instead of separated like this
        #             #     for i in idxs.flatten():
        #             #         x, y = boxes[i][0], boxes[i][1]
        #             #         w, h = boxes[i][2], boxes[i][3]
        #             #         vehicle_box = frame[y:y+h, x:x+w]
        #             #         try:
        #             #             predictions = get_prediction(vehicle_box)
        #             #             print(predictions)
        #             #         except AssertionError:
        #             #             pass
        #         else:
        #             # frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS.show_time, FLAGS.confidence, FLAGS.threshold, boxes, confidences, classids, idxs, infer=False)
        #             # detections = detect_objects(yolo_net, yolo_labels, yolo_layers, yolo_colors, frame, FLAGS.show_time, FLAGS.confidence, FLAGS.threshold)
        #             draw_detections(frame, detections)
        #             count = (count + 1) % 6

                    
        #         if writer is None:
        #             # Initialize the video writer
        #             fourcc = cv.VideoWriter_fourcc(*"MJPG")
        #             writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, 
        #                             (frame.shape[1], frame.shape[0]), True)

        #         writer.write(frame)

        #     print ("[INFO] Cleaning up...")
        #     writer.release()
        #     vid.release()

