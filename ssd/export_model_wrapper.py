import torch
from torch import nn

class ExportModelWrapper(nn.Module):

    def __init__(self, ssd_model):
        super().__init__()
        self._ssd_model = ssd_model
        self.input_names = ['data']
        self.output_names = ['logits', 'deltas']

    def forward(self, x):

        x = self._ssd_model.normalizer(x)
        features = self._ssd_model.backbone(x)
        features = [features[f] for f in self._ssd_model.in_features]

        pred_logits, pred_anchor_deltas = self._ssd_model.head(features)
        pred_logits = [torch.permute(p, dims=(0, 2, 3, 1)) for p in pred_logits]
        pred_anchor_deltas = [torch.permute(p, dims=(0, 2, 3, 1)) for p in pred_anchor_deltas]
        pred_logits = [p.reshape(1, -1) for p in pred_logits]
        pred_anchor_deltas = [p.reshape(1, -1) for p in pred_anchor_deltas]

        pred_anchor_deltas = torch.cat(pred_anchor_deltas, dim=1)

        pred_logits = torch.cat(pred_logits, dim=1)
        pred_logits = pred_logits.reshape(1, -1, self._ssd_model.num_classes + 1)
        pred_logits = torch.softmax(pred_logits, dim=2)
        pred_logits = pred_logits.reshape(1, -1)

        return pred_logits, pred_anchor_deltas
