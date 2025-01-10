import cv2
import tensorflow as tf
import tensorflow_hub as hub

detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")

def detect_notes_on_single_image(image_path, threshold=0.5):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (320, 320))
    img_tensor = tf.convert_to_tensor(img_resized, dtype=tf.uint8)
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    result = detector(img_tensor)
    boxes = result["detection_boxes"].numpy()[0]
    scores = result["detection_scores"].numpy()[0]

    valid_boxes = boxes[scores > threshold]

    return valid_boxes, img