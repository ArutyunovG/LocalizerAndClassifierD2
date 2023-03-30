from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.config import configurable

from detectron2.structures.boxes import pairwise_iou
from detectron2.modeling.matcher import Matcher

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

__all__ = ["LocalizerAndClassifier"]

@META_ARCH_REGISTRY.register()
class LocalizerAndClassifier(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        detector_input_size,
        classifier_input_size,
        roi_w_offset,
        roi_h_offset,
        detector_net,
        classifier_net,
        detector_loss_weight,
        classifier_loss_weight):

        super().__init__()

        self.detector_input_size = detector_input_size
        self.classifier_input_size = classifier_input_size

        self.roi_w_offset = roi_w_offset
        self.roi_h_offset = roi_h_offset

        self.detector_loss_weight = detector_loss_weight
        self.classifier_loss_weight = classifier_loss_weight

        self.detector_trainable = detector_net
        self.detector_stable = copy.deepcopy(self.detector_trainable)

        self.classifier = classifier_net

        self.training = detector_net.training

        self.box_matcher = Matcher([0.3, 0.7], [0, -1, 1])


    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        self.detector_stable.eval()
        detector_input = self.batched_inputs_for_detector(batched_inputs)
        losses = {
                    loss_key: self.detector_loss_weight * loss_value 
                    for (loss_key, loss_value) in self.detector_trainable(detector_input).items()
                 }
        self.detector_stable.load_state_dict(self.detector_trainable.state_dict())
        detector_predictions = self.detector_stable.forward(detector_input)
        self.rescale_detector_predictions(detector_predictions, batched_inputs)
        classifier_input = self.generate_classifier_input(batched_inputs, detector_predictions)
        losses.update(self.calc_classification_loss(classifier_input))
        return losses


    @classmethod
    def from_config(cls, cfg):

        from pydoc import locate

        detector_type = locate(cfg.MODEL.LAC.DETECTOR.MODULE + '.' + cfg.MODEL.LAC.DETECTOR.TYPE)
        detector = detector_type(cfg.MODEL.LAC.DETECTOR.CFG)

        classifier_type = locate(cfg.MODEL.LAC.CLASSIFIER.MODULE + '.' + cfg.MODEL.LAC.CLASSIFIER.TYPE)
        classifier = classifier_type(*cfg.MODEL.LAC.CLASSIFIER.ARGS, **cfg.MODEL.LAC.CLASSIFIER.KWARGS) 
        return {
            "detector_input_size":  cfg.MODEL.LAC.DETECTOR.INPUT_SIZE,
            "classifier_input_size":  cfg.MODEL.LAC.CLASSIFIER.INPUT_SIZE,
            "roi_w_offset": cfg.MODEL.LAC.ROI_W_OFFSET,
            "roi_h_offset": cfg.MODEL.LAC.ROI_H_OFFSET,     
            "detector_net": detector,
            "classifier_net": classifier,
            "detector_loss_weight": cfg.MODEL.LAC.DETECTOR.LOSS_WEIGHT,
            "classifier_loss_weight": cfg.MODEL.LAC.CLASSIFIER.LOSS_WEIGHT
        }


    def generate_classifier_input(self, batched_inputs, detector_predictions):
        classifier_input = {
            'img': [], 'label': []
        }
        for bi, pred in zip(batched_inputs, detector_predictions):
            gt_boxes = bi['instances'].gt_boxes
            pred = pred['instances']
            pred_boxes = pred.pred_boxes.to(gt_boxes.device)
            iou = pairwise_iou(gt_boxes, pred_boxes)
            gt_indices, pred_box_labels = self.box_matcher(iou)
            pred_boxes.tensor = pred_boxes.tensor[pred_box_labels > 0]
            gt_indices = gt_indices[pred_box_labels > 0]

            for gt_index, box in zip(gt_indices, pred_boxes):
                x1, y1, x2, y2 = [round(box[i].item()) for i in range(box.shape[0])]
                w = x2 - x1
                h = y2 - y1
                if w * h == 0:
                    continue
                x1 = round(max(x1 - self.roi_w_offset * w, 0))
                y1 = round(max(y1 - self.roi_h_offset * h, 0))
                x2 = round(min(x2 + self.roi_w_offset * w, bi['image'].shape[2] - 1))
                y2 = round(min(y2 + self.roi_h_offset * h, bi['image'].shape[1] - 1))
                if (x2 - x1) * (y2 - y1) == 0:
                    continue
                roi = bi['image'][:, y1: y2, x1: x2].unsqueeze(0)
                roi = F.interpolate(roi, size=self.classifier_input_size)
                cls = bi['instances'].gt_classes[gt_index].item()
                classifier_input['img'].append(roi.to(self.target_dtype))
                classifier_labels = self.to_classifier_labels(cls)
                assert isinstance(classifier_labels, (dict, list, torch.Tensor)), type(classifier_labels)

                if isinstance(classifier_labels, dict):
                    classifier_input['label'].append(dict())
                    for key in classifier_labels:
                        classifier_input['label'][-1][key] = classifier_labels[key]
                elif isinstance(classifier_labels, list):
                    classifier_input['label'].append(list())
                    for classifier_label in classifier_labels:
                        classifier_input['label'][-1].append(classifier_label)
                elif isinstance(classifier_labels, torch.Tensor):
                    classifier_input['label'].append(classifier_labels)
        
        if not classifier_input['img']:
            return None
        else:
            assert len(classifier_input['img']) == len(classifier_input['label'])
        return classifier_input


    def calc_classification_loss(self, classifier_input):

        if classifier_input is None:
            if hasattr(self, 'classifier_loss_names'):
                return {
                    loss_name: torch.tensor(0)
                    for loss_name in self.classifier_loss_names
                }
            else:
                return dict()

        imgs = torch.cat([img for img in classifier_input['img']], dim=0).to(self.target_device)

        probs = self.classifier(imgs)

        assert isinstance(probs, (dict, list, torch.Tensor)), type(probs)

        loss_names = []
        loss_values = []
        if isinstance(probs, dict):
            for key in probs:
                assert isinstance(key, str), key
                assert isinstance(probs[key], torch.Tensor), probs[key]
                loss_names.append(key)
                labels = torch.cat([torch.tensor(label[key]).view(-1)
                                    for label in classifier_input['label']], dim=0).to(self.target_device)
                loss_values.append(F.cross_entropy(probs[key], labels))
        elif isinstance(probs, list):
            for idx, tensor in enumerate(probs):
                assert isinstance(tensor, torch.Tensor), tensor
                loss_names.append(str(idx))
                labels = torch.cat([torch.tensor(label[idx]).view(-1)
                                    for label in classifier_input['label']], dim=0).to(self.target_device)
                loss_values.append(F.cross_entropy(tensor, labels))
        elif isinstance(probs, torch.Tensor):
            loss_names.append('')
            labels = torch.cat([t for t in classifier_input['label']], dim = 0).view(-1)
            loss_values.append(F.cross_entropy(probs, labels))

        assert len(loss_names) == len(loss_values)
        losses = {
            f'loss_classifier/loss_{name}': self.classifier_loss_weight * value 
            for (name, value) in zip(loss_names, loss_values)
        }

        if hasattr(self, 'classifier_loss_names'):
            assert set(losses.keys()) == set(self.classifier_loss_names)
        else:
            self.classifier_loss_names = losses.keys()

        return losses


    def inference(self, batched_inputs):

        detection_results = self.detector_trainable(batched_inputs)
        results = copy.deepcopy(detection_results)
        for bi, result in zip(batched_inputs, results):
            img = bi['image']
            img = img.unsqueeze(0)
            img = F.interpolate(img, size=(bi['height'], bi['width']))
            img = img.to(self.target_device)
            result['instances'].pred_boxes.clip(img.shape[2:])
            for box_idx, box in enumerate(result['instances'].pred_boxes):
                x1, y1, x2, y2 = [round(box[i].item()) for i in range(box.shape[0])]
                w = x2 - x1
                h = y2 - y1
                if w * h == 0:
                    continue
                x1 = round(max(x1 - self.roi_w_offset * w, 0))
                y1 = round(max(y1 - self.roi_h_offset * h, 0))
                x2 = round(min(x2 + self.roi_w_offset * w, img.shape[3] - 1))
                y2 = round(min(y2 + self.roi_h_offset * h, img.shape[2] - 1))
                if (x2 - x1) * (y2 - y1) == 0:
                    continue
                roi = img[:, :, y1: y2, x1: x2]
                roi = F.interpolate(roi, size=self.classifier_input_size)
                roi = roi.to(torch.float32)
                probs = self.classifier(roi)
                lbl = torch.argmax(probs).cpu().item()
                result['instances'].pred_classes[box_idx] = lbl
        return results


    def batched_inputs_for_detector(self, batch_inputs):
        batch_inputs_for_detector = copy.deepcopy(batch_inputs)
        for bi in batch_inputs_for_detector:
            if bi['image'].shape[1:] != self.detector_input_size:
                image_tensor = bi['image']
                image_tensor = image_tensor.unsqueeze(0)
                image_tensor = F.interpolate(image_tensor, self.detector_input_size)
                image_tensor = image_tensor.squeeze(0)
                bi['image'] = image_tensor
                scale_y = self.detector_input_size[0] / bi['height']
                scale_x = self.detector_input_size[1] / bi['width']
                bi['height'], bi['width'] = self.detector_input_size
            else:
                scale_x = scale_y = 1.0
            for cls_idx in range(len(bi['instances'].gt_classes)):
                bi['instances'].gt_classes[cls_idx] = 0
            bi['instances']._image_size = self.detector_input_size
            bi['instances'].gt_boxes.scale(scale_x=scale_x, scale_y=scale_y)
        return batch_inputs_for_detector

    def to_classifier_labels(self, cls):
        return torch.tensor(cls).to(dtype=torch.int64,
                                    device=self.target_device).view(-1)

    def rescale_detector_predictions(self, detector_predictions, batched_inputs):
        assert len(batched_inputs) == len(detector_predictions)
        for batch in range(len(detector_predictions)):
            scale_y = batched_inputs[batch]['height'] / self.detector_input_size[0]
            scale_x = batched_inputs[batch]['width'] / self.detector_input_size[1]
            detector_predictions[batch]['instances']._image_size = (batched_inputs[batch]['height'],
                                                                    batched_inputs[batch]['width'])
            detector_predictions[batch]['instances'].pred_boxes.scale(scale_x=scale_x, scale_y=scale_y)


    @property
    def target_device(self):
        return list(self.classifier.state_dict().values())[0].device
    
    @property
    def target_dtype(self):
        return list(self.classifier.state_dict().values())[0].dtype
