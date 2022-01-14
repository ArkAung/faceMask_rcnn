"""
    FaceMaskConfig and FaceMaskDataset

    Ref: https://github.com/matterport/Mask_RCNN/blob/master/samples/balloon/balloon.py

"""
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from mrcnn.config import Config
from mrcnn import utils

############################################################
#  Configurations
############################################################


class FaceMaskConfig(Config):
    """Configuration for training on the facemask dataset.
    Derives from the base Config class and overrides some values.

    Consideration to make: Add one big anchor box size to capture facemasks from
    selfie shots (bigger facemasks)
    """
    # Give the configuration a recognizable name
    NAME = "facemask"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + facemask

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Since validation set is quite small
    VALIDATION_STEPS = 10


############################################################
#  Dataset
############################################################

class FaceMaskDataset(utils.Dataset):

    def load_facemask(self, dataset_dir, subset):
        """Load a subset of the FaceMask dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("facemask", 1, "facemask")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "facemask",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a facemask dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "facemask":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "facemask":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def get_facemask_dataset(dataset_path, train=True) -> FaceMaskDataset:
    """

    :param dataset_path: Path of directory which contains both train and val directories
    :param train: True for training dataset, False for validation dataset
    :return:
    """
    dataset = FaceMaskDataset()
    dataset_type = 'train' if train else 'val'
    dataset.load_facemask(dataset_path, dataset_type)
    dataset.prepare()
    return dataset
