import torch
import numpy as np


a = np.array([3.1,4.5])
a = torch.Tensor(a)
y_true_list = [a, a]
y = torch.cat(y_true_list)
print(y)