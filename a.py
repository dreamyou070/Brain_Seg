import torch
from torch.nn import functional as F
masks_pred = torch.randn(1,4,64,64)
predict = F.softmax(masks_pred, dim=1).float()
print(predict.shape)


#F.one_hot(true_masks, segmentation_model.module.n_classes).permute(0, 3, 1, 2).float(),
a = torch.arange(0, 5) % 3
#F.one_hot(torch.arange(0, 5) % 3, num_classes=5)
print(a)