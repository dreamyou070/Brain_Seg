import torch

class_weight = {0: 0.0027217850085457886, 1: 0.22609416133509747, 2: 0.17582554657020089, 3: 0.5953585070861559}
target = torch.ones((1,4,64))
bs = target.shape[0]
class_weight = torch.tensor(list(class_weight.values())).unsqueeze(dim=0).repeat(bs, 1)
target = target * class_weight[:,:,None]
#print(target)

true_mask_one_vector = torch.randn((1,4096))
true_mask_one_vector = true_mask_one_vector .view(64,64)
print(true_mask_one_vector .shape)