"""
    Train Mask RCNN on self-annotated face mask dataset.

    Ref: https://github.com/matterport/Mask_RCNN/blob/master/samples/balloon/balloon.py

Usage: python train.py --dataset /path/where/you/downloaded/dataset --weights "coco"
"""
import os

from facemask import FaceMaskConfig, get_facemask_dataset
from mrcnn.model import MaskRCNN
from mrcnn.utils import download_trained_weights

ROOT_DIR = '.'

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


def train_network(mdl: MaskRCNN, dataset_path: str,
                  lr: float = 1e-3, epochs: int = 5, training_type: str = 'heads') -> None:
    """
    Train Mask RCNN model

    :param mdl: MaskRCNN model to be trained
    :param dataset_path: Path of directory which contains both train and val directories
    :param lr: Learning rate
    :param epochs: Epochs to be trained
    :param training_type: "head|all" head - freeze the body and train only the newly added heads
                                     all - train the whole network
    :return: None
    """
    assert training_type == "heads" or training_type == "all", "Training type should be 'heads' or 'all'"

    dataset_train = get_facemask_dataset(dataset_path=dataset_path, train=True)
    dataset_val = get_facemask_dataset(dataset_path=dataset_path, train=False)

    mdl.train(dataset_train, dataset_val,
              learning_rate=lr,
              epochs=epochs,
              layers=training_type)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect facemasks.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/facemask/dataset/",
                        help='Directory which contains both train '
                             'and val directories of Face Mask dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--epochs', required=False,
                        default=5, help="Training epochs")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--learning_rate', required=False,
                        help='Learning rate to be used in training')
    parser.add_argument('--training_type', help='Training type [heads|all]')
    args = parser.parse_args()

    # Validate that there is training dataset specified
    assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = FaceMaskConfig()
    config.display()

    # Create model
    model = MaskRCNN(mode="training", config=config,
                     model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    if args.learning_rate:
        learning_rate = float(args.learning_rate)
    else:
        learning_rate = config.LEARNING_RATE
    train_network(model, dataset_path=args.dataset, lr=learning_rate,
                  epochs=int(args.epochs), training_type=args.training_type)

