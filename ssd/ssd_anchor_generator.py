from detectron2.modeling.anchor_generator import ANCHOR_GENERATOR_REGISTRY
from detectron2.modeling.anchor_generator import BufferList, DefaultAnchorGenerator

from detectron2.layers import ShapeSpec

import torch
from torch import nn

from typing import List

import math

@ANCHOR_GENERATOR_REGISTRY.register()
class SSDAnchorGenerator(DefaultAnchorGenerator):

    """
    For a set of image sizes and feature maps, computes a set of anchors for SSD models.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):

        nn.Module.__init__(self)

        sizes         = cfg.MODEL.ANCHOR_GENERATOR.SIZES
        aspect_ratios = cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS
        self.strides  = [x.stride for x in input_shape]
        self.offset   = cfg.MODEL.ANCHOR_GENERATOR.OFFSET

        assert 0.0 <= self.offset < 1.0, self.offset

        self.num_features = len(self.strides)
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)


    def generate_cell_anchors(self, sizes, aspect_ratios):
        def add_anchor_with_wh(w, h):
            x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchors_for_curr_level.append([x0, y0, x1, y1])

        anchors_on_all_levels = []

        for size_idx in range(len(sizes) - 1):

            anchors_for_curr_level = []

            min_size = sizes[size_idx]
            max_size = sizes[size_idx + 1]

            assert len(min_size) == 1, min_size
            assert len(max_size) == 1, max_size

            min_size = min_size[0]
            max_size = max_size[0]

            ar_per_level = aspect_ratios[size_idx]

            if 1.0 in ar_per_level:
                add_anchor_with_wh(min_size, min_size)

            sqrt_sz = math.sqrt(min_size * max_size)
            add_anchor_with_wh(sqrt_sz, sqrt_sz)

            eps = 1e-5

            for ar in ar_per_level:

                if abs(ar - 1.0) < eps:
                    continue

                assert ar > 0, ar

                w = min_size * math.sqrt(ar)
                h = min_size / math.sqrt(ar)

                add_anchor_with_wh(w, h)

            anchors_on_all_levels.append(torch.tensor(anchors_for_curr_level))

        return anchors_on_all_levels

    def _calculate_anchors(self, sizes, aspect_ratios):
        # If one size (or aspect ratio) is specified and there are multiple feature
        # maps, then we "broadcast" anchors of that single size (or aspect ratio)
        # over all feature maps.
        if len(sizes) == 1:
            sizes *= self.num_features
        if len(aspect_ratios) == 1:
            aspect_ratios *= self.num_features

        if isinstance(self, SSDAnchorGenerator):
            assert self.num_features == len(sizes) - 1
        else:
            assert self.num_features == len(sizes)
        assert self.num_features == len(aspect_ratios)

        if isinstance(self, SSDAnchorGenerator):
            cell_anchors = self.generate_cell_anchors(sizes, aspect_ratios)
        else:
            cell_anchors = [
                self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
            ]

        return BufferList(cell_anchors)


