import os
import numpy as np

dir = 'class_weights.npy'
weight = np.load(dir)
total_weight = weight.sum()
weight = weight / total_weight
print(total_weight)
class_weight = dict(enumerate(weight))
print(class_weight)

def class_weight_to_sample_weights(y, class_weight):
    sample_weight = np.ones(shape=(y.shape[0],))
    if len(y.shape) > 1:
        if y.shape[-1] != 1:
            y = np.argmax(y, axis=-1)
        else:
            y = np.squeeze(y, axis=-1)
    y = np.round(y).astype("int32")
    for i in range(y.shape[0]):
        sample_weight[i] = class_weight.get(int(y[i]), 1.0) # label by label weight
    return sample_weight

weight_0 = class_weight.get(3,1)
y = np.array([0,1,2])
sample_weight = class_weight_to_sample_weights(y, class_weight)
print(sample_weight)