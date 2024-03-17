import torch
import numpy as np
from keras.metrics import MeanIoU

y_pred = torch.randn(1,4,64,64)
y_pred = y_pred.permute(0,2,3,1).numpy()
y_pred_argmax = np.argmax(y_pred, axis=3)
IOU_keras = MeanIoU(num_classes=4)
IOU_keras.update_state(y_pred_argmax, y_pred_argmax)
print(f"mean IoU = {IOU_keras.result().numpy()}")