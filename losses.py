from __future__ import print_function

import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, CFG, temperature=0.07, contrast_mode='all',):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature #base_temperature
        self.CFG = CFG
        
    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device(self.CFG.device)

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
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask ## (batch_size * anchor_count, batch_size * anchor_count)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask ## 분자 (batch_size * anchor_count, batch_size * anchor_count)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) ## 분모 (batch_size * anchor_count, 1)

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1) #양의 쌍의 수 
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs) # 수가 너무 적으면 1로 수정 
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class FeatureBatchdSupConLoss(nn.Module):
    """Modified Supervised Contrastive Loss without logits_mask."""
    def __init__(self, CFG, temperature=0.07):
        super(FeatureBatchdSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature
        self.CFG = CFG
        self.device = torch.device(self.CFG.device)
    def forward(self, features, contrast_features, labels=None, contrast_labels=None):
        """
        Compute the modified supervised contrastive loss using per-sample contrast features and labels.

        Args:
            features: Anchor features of shape [batch_size, feature_dim].
            contrast_features: Contrast features of shape [batch_size, num_contrast, feature_dim].
            labels: Ground truth labels for the anchors of shape [batch_size].
            contrast_labels: Ground truth labels for the contrast features of shape [batch_size, num_contrast].

        Returns:
            A scalar loss value.
        """
        batch_size = features.shape[0]
        feature_dim = features.shape[1]
        num_contrast = contrast_features.shape[1]

        if labels is None or contrast_labels is None:
            raise ValueError('Both labels and contrast_labels must be provided.')

        # features: [batch_size, feature_dim]
        # contrast_features: [batch_size, num_contrast, feature_dim]
        features_expanded = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        logits = torch.bmm(features_expanded, contrast_features.transpose(1, 2))  # [batch_size, 1, num_contrast]
        logits = logits.squeeze(1)  # [batch_size, num_contrast]

        logits = logits / self.temperature  # [batch_size, num_contrast]

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)  # [batch_size, 1]
        logits = logits - logits_max.detach()

        labels = labels.view(batch_size, 1)  # [batch_size, 1]
        mask = torch.eq(labels, contrast_labels).float().to(self.device)  # [batch_size, num_contrast]

        exp_logits = torch.exp(logits)  # [batch_size, num_contrast]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)  # [batch_size, 1]

        mask_sum = mask.sum(1)  # [batch_size]
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum).to(self.device))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum  # [batch_size]

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    