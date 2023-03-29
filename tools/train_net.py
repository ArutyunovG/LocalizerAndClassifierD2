
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

from localizer_and_classifier import add_lac_config
from ssd import add_ssd_config

import backbone
import datasets

from detectron2.data import build_detection_test_loader, build_detection_train_loader

from dataset_mapper import DatasetMapper

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


class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg, mapper=None):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True))
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, is_train=False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        evaluator_list = []
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        return DatasetEvaluators(evaluator_list)



def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
