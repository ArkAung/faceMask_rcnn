"""
    Inferencing functions and utils
     - Config class for inferencing
     - Building MaskRCNN model form weights path and device type string
     - Processing image which is uploaded
     - Getting predictions (RoI and segmentation coverage)
     - Getting percent coverage of segmented masks over the whole image.
"""
import os
from typing import Dict, Tuple

import numpy as np
import skimage
import tensorflow as tf

from facemask import FaceMaskConfig
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
from mrcnn.visualize import save_masked_image


class InferenceConfig(FaceMaskConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def visualize_masks(image_dir: str, model: MaskRCNN):
    image_paths = []
    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
            image_paths.append(os.path.join(image_dir, filename))

    for image_path in image_paths:
        img = skimage.io.imread(image_path)
        img_arr = np.array(img)
        results = model.detect([img_arr], verbose=0)
        r = results[0]
        display_instances(img, r['rois'], r['masks'], r['class_ids'],
                          ['bg', 'face_mask'], r['scores'], figsize=(5, 5))


def get_predictions(image_path: str, model: MaskRCNN) -> Tuple[Dict, float]:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=0)
    r = results[0]
    coverage = get_mask_coverage(r['masks'])
    return r, coverage


def get_model(weight_path: str, device: str) -> MaskRCNN:
    tf_device = '/cpu:0' if device == 'cpu' else '/gpu:0'
    cfg = InferenceConfig()
    with tf.device(tf_device):
        mdl = MaskRCNN(mode="inference", model_dir='logs',
                       config=cfg)
    mdl.load_weights(weight_path, by_name=True)
    return mdl


def process_image(model: MaskRCNN, img_path: str, result_dir: str, output_filename: str) -> Tuple[Dict, float]:
    r, coverage = get_predictions(img_path, model=model)
    img_arr = skimage.io.imread(img_path)
    output_filepath = os.path.join(result_dir, output_filename)
    save_masked_image(img_arr, r['rois'], r['masks'], r['class_ids'],
                      ['BG', 'Face Mask'], r['scores'], save_filepath=output_filepath)
    return r, coverage


def get_mask_coverage(masks: np.ndarray) -> float:
    mask_pixels = np.sum(masks)
    tot_pixels = masks.shape[0] * masks.shape[1]
    return (mask_pixels / tot_pixels) * 100
