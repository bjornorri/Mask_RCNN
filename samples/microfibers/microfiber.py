"""
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 microfiber.py train --dataset=/path/to/microfiber/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 microfiber.py train --dataset=/path/to/microfiber/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 microfiber.py train --dataset=/path/to/microfiber/dataset --weights=imagenet

    # Apply color splash to an image
    python3 microfiber.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 microfiber.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import datetime
import subprocess
import random
import numpy as np
import skimage.draw

# Hack CUDA_VISIBLE_DEVICES to always find available GPUs
cmd = "nvidia-smi | awk '{print $2}' | awk '/Processes:/{y=1;next}y' | awk '/GPU/{y=1;next}y'"
output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()

used_gpus = set([int(s) for s in output.split('\n')])
all_gpus = set(range(8))
available_gpus = all_gpus - used_gpus

if len(available_gpus) == 0:
    print('No GPUs!')
    sys.exit(0)

print('Found GPUs: {}'.format(','.join([str(gpu) for gpu in available_gpus])))

# Leave some available
gpus = list(available_gpus)
max_gpus = 3
if len(gpus) > max_gpus:
    gpus = gpus[:max_gpus]

# Set the environment variable
visible_devices = ','.join([str(i) for i in gpus])
os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

print('Using GPUs: {}'.format(visible_devices))

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import visualize


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# NUM_SAMPLES = 2821
NUM_SAMPLES = 427
TRAIN_SPLIT = 1.0
RESULTS_DIR = os.path.join(ROOT_DIR, "/local/home/bjornorri/results/microfibers")


############################################################
#  Configurations
############################################################


class MicrofiberConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "microfiber"

    GPU_COUNT = len(gpus)

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + microfiber

    num_samples = NUM_SAMPLES
    # num_samples = 200
    # Number of training steps per epoch
    EPOCHS = 200
    STEPS_PER_EPOCH = int(TRAIN_SPLIT * num_samples)
    # STEPS_PER_EPOCH = 100

    # VALIDATION_STEPS = min(num_samples - STEPS_PER_EPOCH, 10)
    VALIDATION_STEPS = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    BACKBONE = "resnet50"

    # Can't afford mini-mask since the fibers are small
    # USE_MINI_MASK = False

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Reduce number of units in classification layer
    # FPN_CLASSIF_FC_LAYERS_SIZE = 256

    # Faster learning rate
    LEARNING_RATE = 0.001
    # LEARNING_RATE = 0.1

    # RPN_NMS_THRESHOLD = 0.99
    # DETECTION_NMS_THRESHOLD = 0.99


############################################################
#  Dataset
############################################################

class MicrofiberDataset(utils.Dataset):

    def load_microfiber(self, dataset_dir, subset):
        """Load a subset of the Microfiber dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("microfiber", 1, "microfiber")

        # Get the sample names and paths.
        samples = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        sample_paths = [os.path.join(dataset_dir, s) for s in samples]

        random.Random(37).shuffle(sample_paths)
        sample_paths = sample_paths[:NUM_SAMPLES]

        index = int(TRAIN_SPLIT * len(sample_paths))

        # if subset == 'train':
            # sample_paths = sample_paths[:index]
        # elif subset == 'val':
            # sample_paths = sample_paths[index:]

        # Add each image.
        for path in sample_paths:
            name = path.split('/')[-1]
            image_path = os.path.join(path, 'image.png')
            self.add_image("microfiber", image_id=name, path=image_path)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a microfiber dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "microfiber":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask_dir = os.path.join(os.path.dirname(info["path"]), 'masks')
        mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]
        masks = [skimage.io.imread(f) for f in mask_files]

        # Generate an empty mask if no masks are found.
        if len(masks) == 0:
            mask = np.zeros((512, 512))
            masks.append(mask)

        masks = np.array(masks)
        masks = np.moveaxis(masks, 0, -1)

        return masks.astype(np.bool), np.ones([masks.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "microfiber":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = MicrofiberDataset()
    dataset_train.load_microfiber(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MicrofiberDataset()
    dataset_val.load_microfiber(args.dataset, "val")
    dataset_val.prepare()

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
                # learning_rate=config.LEARNING_RATE,
                # epochs=1,
                # layers='heads')

    print("Training all layers")
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=config.EPOCHS,
                layers="all")


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

############################################################
#  Segmentation
############################################################


def segment(model, dataset_dir, stats=True):
    print("Segmenting images")

    if stats:
        def iou(a, b):
            intersection = (np.logical_and(a, b) == 1).sum()
            union = (np.logical_or(a, b) == 1).sum()
            return intersection / union

        total_fibers = 0
        total_found = 0
        iou_values = []

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = MicrofiberDataset()
    dataset.load_microfiber(dataset_dir, 'val')
    dataset.prepare()
    # Load over images
    submission = []
    for n, image_id in enumerate(dataset.image_ids):
        print("Image {} / {}".format(n + 1, len(dataset.image_ids)))
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

        if stats:
            pred_masks = np.moveaxis(r["masks"], -1, 0)
            true_masks = np.moveaxis(dataset.load_mask(image_id)[0], -1, 0)

            # Calculate metrics
            # Keep track of fibers that were detected
            found_mask_idx = set()

            for i, pmask in enumerate(pred_masks):
                match = max(true_masks, key=lambda x: iou(pmask, x))
                found_mask_idx.add(i)
                val = iou(pmask, match)
                iou_values.append(val)

            total_fibers += len(true_masks)
            total_found += len(found_mask_idx)

    # Output metrics
    if stats:
        recall = total_found / total_fibers
        mean_iou = np.mean(iou_values)
        print("Recall: {}".format(recall))
        print("Mean IoU: {}".format(mean_iou))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect microfibers.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/microfiber/dataset/",
                        help='Directory of the Microfiber dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "segment":
        assert args.dataset

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MicrofiberConfig()
    else:
        class InferenceConfig(MicrofiberConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
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

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "segment":
        segment(model, dataset_dir=args.dataset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'segment'".format(args.command))
