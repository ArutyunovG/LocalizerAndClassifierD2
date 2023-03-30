import argparse
import os
import shutil

import torch
from torch import nn

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


import backbone

from localizer_and_classifier import add_lac_config
from ssd import add_ssd_config
from ssd.export_model_wrapper import ExportModelWrapper

import onnx
from onnxsim import simplify

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ssd_config(cfg)
    add_lac_config(cfg)
    # have to provide domain specific data on tool level
    cfg.INPUT.COCO_VAL_PATH = ''
    cfg.MODEL.LAC.CLASSIFIER.KWARGS.pretrained = False
    cfg.MODEL.LAC.CLASSIFIER.KWARGS.num_classes = 1
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output",
        default="export_onnx",
        metavar="FILE",
        help="path to onnx export folder",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def export_simplified_model(model_output_path,
                            model,
                            input_size,
                            input_names,
                            output_names,
                            input_channels=3):
    h, w = input_size
    torch.onnx.export(model,
                    torch.randn(1, input_channels, h, w, requires_grad=True),
                    model_output_path,
                    input_names = input_names, 
                    output_names = output_names)
    onnx_model = onnx.load(model_output_path)
    onnx_model_simp, check = simplify(onnx_model)
    assert check
    onnx.save(onnx_model_simp, model_output_path)


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)

    model = build_model(cfg)
    model = model.to(torch.device('cpu'))
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    classifier_model = model.classifier
    classifier_model.eval()
    ch, cw = cfg.MODEL.LAC.CLASSIFIER.INPUT_SIZE
    export_simplified_model(os.path.join(args.output, 'classifier.onnx'),
                            classifier_model,
                            (ch, cw),
                            input_names=['data'],
                            output_names=['probs'])


    detector_model = ExportModelWrapper(model.detector_stable)
    detector_model.eval()
    dh, dw = cfg.MODEL.LAC.DETECTOR.INPUT_SIZE
    export_simplified_model(os.path.join(args.output, 'detector.onnx') ,
                            detector_model,
                            (dh, dw),
                            input_names=detector_model.input_names,
                            output_names=detector_model.output_names)

