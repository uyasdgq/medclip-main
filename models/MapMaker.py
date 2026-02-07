
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MapMaker(nn.Module):

    def __init__(self, image_size, num_layers=3):

        super(MapMaker, self).__init__()
        self.image_size = image_size
        # Learnable weights for multi-layer feature fusion
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)


    def forward(self, vision_adapter_features, propmt_adapter_features):
        anomaly_maps = []

        for i, vision_adapter_feature in enumerate(vision_adapter_features):
            B, H, W, C = vision_adapter_feature.shape
            anomaly_map = (vision_adapter_feature.view((B, H * W, C)) @ propmt_adapter_features).contiguous().view(
                (B, H, W, -1)).permute(0, 3, 1, 2)

            anomaly_maps.append(anomaly_map)

        # Learnable weighted fusion instead of simple mean
        weights = F.softmax(self.layer_weights, dim=0)
        anomaly_map = sum(w * m for w, m in zip(weights, anomaly_maps))
        anomaly_map = F.interpolate(anomaly_map, (self.image_size, self.image_size), mode='bilinear', align_corners=True)
        return torch.softmax(anomaly_map, dim=1)