def collect_anomal_map_loss_multi_crossentropy(self,
                                               attn_score,  # [8, 64*64, 4]
                                               gt_vector,  # [64*64]
                                               do_normal_activating=True):

    gt_vector = gt_vector.squeeze().type(torch.LongTensor).to(attn_score.device)  # [res*res]
    loss = self.multiclassification_loss_fn(attn_score, gt_vector)  # what form ?
    self.anomal_map_loss.append(loss)

import torch
from torch import nn

multiclassification_loss_fn = nn.CrossEntropyLoss()

attn_score = torch.randn(8,64*64,4)
attn_score = attn_score.squeeze()  # [8,res*res,4]
attn_score = attn_score.mean(dim=0)  # [res*res,4]

gt_vector = torch.randn(64*64,1) * 0
gt_vector = gt_vector.squeeze().type(torch.LongTensor).to(attn_score.device)  # [res*res]
loss = multiclassification_loss_fn(attn_score, gt_vector)  # what form ?
print(loss)
map_loss = torch.stack([loss], dim=0)
map_loss = map_loss.mean()
print(map_loss)
