import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """
    Reference: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, num_classes=6625, feat_dim=96, center_file_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.randn(self.num_classes, self.feat_dim, dtype=torch.float64)

        if center_file_path is not None:
            assert os.path.exists(
                center_file_path
            ), f"center path({center_file_path}) must exist when it is not None."
            with open(center_file_path, "rb") as f:
                char_dict = pickle.load(f)
                for key in char_dict.keys():
                    self.centers[key] = torch.tensor(char_dict[key], dtype=torch.float64)

    def forward(self, predicts, batch):
        assert isinstance(predicts, (list, tuple))
        features, predicts = predicts

        feats_reshape = features.view(-1, features.shape[-1]).to(torch.float64)
        label = torch.argmax(predicts, dim=2)
        label = label.view(-1)

        batch_size = feats_reshape.shape[0]

        # calc l2 distance between feats and centers
        square_feat = torch.sum(feats_reshape ** 2, dim=1, keepdim=True)
        square_feat = square_feat.expand(batch_size, self.num_classes)

        square_center = torch.sum(self.centers ** 2, dim=1, keepdim=True)
        square_center = square_center.expand(self.num_classes, batch_size).to(torch.float64)
        square_center = square_center.t()

        distmat = square_feat + square_center
        feat_dot_center = torch.matmul(feats_reshape, self.centers.t())
        distmat = distmat - 2.0 * feat_dot_center

        # generate the mask
        classes = torch.arange(self.num_classes, dtype=torch.int64)
        label = label.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = (classes.expand(batch_size, self.num_classes) == label).to(torch.float64)
        dist = distmat * mask

        loss = torch.sum(torch.clamp(dist, min=1e-12, max=1e12)) / batch_size
        return {"loss_center": loss}