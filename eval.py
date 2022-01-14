"""
    Eval computes mAP @ IOU = 0.5

"""
import numpy as np
from mrcnn import utils
from mrcnn.model import load_image_gt, mold_image, MaskRCNN
from facemask import FaceMaskConfig
from facemask import get_facemask_dataset
import tensorflow as tf
import time

class InferenceConfig(FaceMaskConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def eval_on_val(dataset_path: str, model: MaskRCNN, config: InferenceConfig):
    val_dataset = get_facemask_dataset(dataset_path, train=False)

    APs = []
    elapsed_times = []
    for image_id in val_dataset.image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(val_dataset, config,
                                                                         image_id, use_mini_mask=False)
        molded_images = np.expand_dims(mold_image(image, config), 0)
        # Run object detection
        start_time = time.time()
        results = model.detect([image], verbose=0)
        elapsed = time.time() - start_time
        elapsed_times.append(elapsed)
        r = results[0]
        # Compute AP
        ap, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r["rois"], r["class_ids"], r["scores"], r['masks'],
                                                             iou_threshold=0.5)
        APs.append(ap)

    print("mAP: ", np.mean(APs))
    print("Mean elapsed time: ", np.mean(elapsed_times))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate trained Mask R-CNN on validation set.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/facemask/dataset/",
                        help='Directory which contains both train '
                             'and val directories of Face Mask dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default='logs/',
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--device', required=False,
                        default='cpu',
                        help="Device to run evaluation on [cpu(default)|gpu]")
    args = parser.parse_args()

    assert args.device == 'cpu' or args.device == 'gpu', "Device should either be 'cpu' or 'gpu'"
    device = '/cpu:0' if args.device == 'cpu' else '/gpu:0'

    cfg = InferenceConfig()

    with tf.device(device):
        mdl = MaskRCNN(mode="inference", model_dir=args.logs,
                       config=cfg)
    mdl.load_weights(args.weights, by_name=True)
    eval_on_val(dataset_path=args.dataset, model=mdl, config=cfg)
