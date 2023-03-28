
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

from localizer_and_classifier import add_lac_config
from ssd import add_ssd_config

import backbone
import datasets

from detectron2.data.build import get_detection_dataset_dicts, build_batch_data_loader
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import RepeatFactorTrainingSampler, TrainingSampler
import logging

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
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
        )
        dataset = DatasetFromList(dataset_dicts, copy=False)

        if mapper is None:
            mapper = DatasetMapper(cfg, True)
        dataset = MapDataset(dataset, mapper)

        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))
        return build_batch_data_loader(
            dataset,
            sampler,
            cfg.SOLVER.IMS_PER_BATCH,
            aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )
    

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
