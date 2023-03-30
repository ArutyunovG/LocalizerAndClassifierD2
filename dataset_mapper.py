import copy
import os
import random

import torch

import cv2
import numpy as np

from detectron2.data import detection_utils as utils
from detectron2.structures import instances

__all__ = ["DatasetMapper"]

class DatasetMapper:


    def __init__(self, cfg, is_train=True):

        assert not cfg.MODEL.MASK_ON
        assert not cfg.MODEL.KEYPOINT_ON
        assert not cfg.MODEL.LOAD_PROPOSALS

        self.is_train = is_train

        self.img_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT

        self.target_size = cfg.INPUT.MIN_SIZE_TRAIN
        background_images_folder = cfg.INPUT.COCO_VAL_PATH

        if not background_images_folder:
            self.background_images = None
        else:
            self.background_images = [os.path.join(background_images_folder, x)
                                      for x in os.listdir(background_images_folder)]


    def __call__(self, dataset_dict):

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        if not self.is_train:
            dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
            dataset_dict.pop("annotations", None)
            return dataset_dict

        out_size = random.randint(256, 1024)
        min_size = 128

        prescale = random.uniform(0.5, 2.0)
        image = cv2.resize(image, tuple([0, 0]), fx=prescale, fy=prescale)
        image = cv2.resize(image, tuple([0, 0]), fx=1.0/prescale, fy=1.0/prescale)

        image_size_w = random.randint(min_size, out_size)

        ar_offset_eps = 0.3
        ar_offset = random.uniform(1.0 - ar_offset_eps, 1.0 + ar_offset_eps)
        ar_coeff = ar_offset * (image.shape[0] / image.shape[1])
        image_size_h = min(round(ar_coeff * image_size_w), out_size)

        scale_x, scale_y = image_size_w / image.shape[1], image_size_h / image.shape[0]        
        image = cv2.resize(image, (image_size_w, image_size_h))

        for anno in dataset_dict['annotations']:
            anno['bbox'][0] *= scale_x
            anno['bbox'][1] *= scale_y
            anno['bbox'][2] *= scale_x
            anno['bbox'][3] *= scale_y
            for segm in anno['segmentation']:
                for ci in range(0, len(segm), 2):
                    segm[ci + 0] *= scale_x
                    segm[ci + 1] *= scale_y

        assert self.background_images

        if random.randint(0, 1):
            background_img_path = random.choice(self.background_images)
            background_img = cv2.imread(background_img_path, cv2.IMREAD_COLOR)
            background_img = cv2.resize(background_img, (out_size, out_size))
        else:
            background_img = np.ones_like(image) * random.randint(0, 255)
            background_img = cv2.resize(background_img, (out_size, out_size))


        background_img[:image_size_h, :image_size_w, :] = image
        image = background_img

        dataset_dict['width'] = out_size
        dataset_dict['height'] = out_size

        utils.check_image_size(dataset_dict, image)

        assert "annotations"  in dataset_dict

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annos = dataset_dict["annotations"]

        instances = utils.annotations_to_instances(
            annos, image.shape[:2], mask_format=self.mask_format
        )

        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

