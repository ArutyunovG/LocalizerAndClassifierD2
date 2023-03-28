from detectron2.data.datasets import register_coco_instances

register_coco_instances("train", {}, "/home/grisha/piece_train/dataset/dataset/annotations/train.json", "/home/grisha/piece_train/dataset/dataset/train/")
register_coco_instances("val", {}, "/home/grisha/piece_train/dataset/dataset/annotations/train.json", "/home/grisha/piece_train/dataset/dataset/train/")
