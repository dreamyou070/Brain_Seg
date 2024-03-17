from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
score = confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
actual_axis, pred_axis = score.shape
for actual_idx in range(actual_axis) :
    total_actual_num = score[actual_idx]
    total_actual_num = sum(total_actual_num)
    precision = score[actual_idx, actual_idx] / total_actual_num
    print(precision)