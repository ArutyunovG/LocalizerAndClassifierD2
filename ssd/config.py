from detectron2.config import CfgNode as CN


def add_ssd_config(cfg):
    """
    Add config for SSD
    """
    _C = cfg

    # ---------------------------------------------------------------------------- #
    # SSD Head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.SSD = CN()

    # This is the number of foreground classes.
    _C.MODEL.SSD.NUM_CLASSES = 20

    _C.MODEL.SSD.IN_FEATURES = [
        "relu4_3",
        "relu7",
        "conv6_2_relu",
        "conv7_2_relu",
        "conv8_2_relu",
        "conv9_2_relu",
    ]

    # Convolutions to use in the cls and bbox tower
    # NOTE: this doesn't include the last conv for logits
    _C.MODEL.SSD.NUM_CONVS = 0

    # IoU overlap ratio [bg, fg] for labeling anchors.
    # Anchors with < bg are labeled negative (0)
    # Anchors  with >= bg and < fg are ignored (-1)
    # Anchors with >= fg are labeled positive (1)
    _C.MODEL.SSD.IOU_THRESHOLDS = [0.4, 0.5]
    _C.MODEL.SSD.IOU_LABELS = [0, -1, 1]

    # Prior prob for rare case (i.e. foreground) at the beginning of training.
    # This is used to set the bias for the logits layer of the classifier subnet.
    # This improves training stability in the case of heavy class imbalance.
    _C.MODEL.SSD.PRIOR_PROB = 0.01

    # Inference cls score threshold, only anchors with score > INFERENCE_TH are
    # considered for inference (to improve speed)
    _C.MODEL.SSD.SCORE_THRESH_TEST = 0.05
    _C.MODEL.SSD.TOPK_CANDIDATES_TEST = 1000
    _C.MODEL.SSD.NMS_THRESH_TEST = 0.5

    # Weights on (dx, dy, dw, dh) for normalizing SSD anchor regression targets
    _C.MODEL.SSD.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

    # Loss parameters

    # Allowed values for loss name are "Multibox" or "Focal"
    # "Multibox" loss is MultiboxLossLayer from Caffe with MAX_NEGATIVE mining_type
    # "Focal" stands for Focal loss
    _C.MODEL.SSD.LOSS_NAME="Multibox"
    _C.MODEL.SSD.FOCAL_LOSS_GAMMA = 2.0
    _C.MODEL.SSD.FOCAL_LOSS_ALPHA = 0.25
    # Maximum value of neg / pos anchors fraction
    _C.MODEL.SSD.MULTIBOX_LOSS_NEG_POS_RATIO=3.0
    # Box loss is always SmoothL1, regardless of conf loss
    _C.MODEL.SSD.SMOOTH_L1_LOSS_BETA = 1.0

    _C.INPUT.COCO_VAL_PATH = None
    _C.MODEL.BACKBONE.PRETRAINED = False
