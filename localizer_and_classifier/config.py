from detectron2.config import CfgNode as CN

import copy

def add_lac_config(cfg):
    _C = cfg

    _C.MODEL.BACKBONE.PRETRAINED = False

    _C.MODEL.LAC = CN()
    
    _C.MODEL.LAC.DETECTOR = CN()
    _C.MODEL.LAC.DETECTOR.MODULE = ''
    _C.MODEL.LAC.DETECTOR.TYPE = ''
    _C.MODEL.LAC.DETECTOR.CFG = copy.deepcopy(cfg)

    _C.MODEL.LAC.CLASSIFIER = CN()
    _C.MODEL.LAC.CLASSIFIER.MODULE = ''
    _C.MODEL.LAC.CLASSIFIER.TYPE = ''
    _C.MODEL.LAC.CLASSIFIER.INPUT_SIZE = (32, 32)
    _C.MODEL.LAC.CLASSIFIER.ARGS = tuple()
    _C.MODEL.LAC.CLASSIFIER.KWARGS = CN()

    _C.MODEL.LAC.ROI_W_OFFSET = 0.0
    _C.MODEL.LAC.ROI_H_OFFSET = 0.0
