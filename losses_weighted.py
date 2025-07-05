from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

# class SupConLoss_weight(nn.Module):
class SupConLoss_weight(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, CFG, temperature=0.07, contrast_mode='all', class_weights=None):
        super(SupConLoss_weight, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature
        self.CFG = CFG
        self.class_weights = class_weights

    def forward(self, feature1, feature2, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            feature1: hidden vector of shape [bsz, ...].
            feature2: hidden vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)
        features = torch.cat([feature1.unsqueeze(1), feature2.unsqueeze(1)], dim=1)
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Debugging: Print shapes and indices
        # print("Batch size:", batch_size)
        # print("Anchor count:", anchor_count)
        # print("Mask shape:", mask.shape)
        # print("Logits shape:", logits.shape)
        
        # mask-out self-contrast cases
        logits_mask = torch.ones_like(mask).to(device)
        indices = torch.arange(batch_size * anchor_count).view(-1, 1).to(device)
        
        # Validate indices
        max_index = logits_mask.size(1) - 1
        if torch.any(indices > max_index):
            raise ValueError("Indices out of bounds in scatter operation.")
        
        logits_mask.scatter_(1, indices, 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Apply class weights if provided
        if self.class_weights is not None and labels is not None:
            class_weights = self.class_weights.to(device)
            # Ensure labels are within range
            if torch.any(labels >= len(class_weights)):
                raise ValueError("Label index out of range for class weights.")
            weights = class_weights[labels.view(-1)]
            weights = weights.repeat(anchor_count)  # Make sure weights match the size of mean_log_prob_pos
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos * weights
        else:
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size).mean()

        return loss