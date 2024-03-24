import numpy as np





confusion_matrix = np.array([[12656660, 19398,5828],
[37853, 563654, 17585],
[10269, 14814, 43283],])


real_axis, pred_axis = confusion_matrix.shape
values = []
total_nums = []
for r_idx in range(real_axis) :
    predict_result = confusion_matrix[r_idx]
    total_num = predict_result.sum()
    total_nums.append(total_num)
    normalized_value = [round(value/total_num,3) for value in predict_result]
    values.append(normalized_value)
values = np.array(values)
print(values)
print(f'\n classwise total sample = {total_nums}')
"""
import torch
import torch.nn.functional as F
masks_pred = torch.randn(1,3,256,256)
y_pred = torch.argmax(masks_pred, dim=1).flatten() # change to one_hot
y_pred = F.one_hot(y_pred, num_classes=3)
print(y_pred.shape)
"""