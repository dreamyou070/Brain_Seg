from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch


a = torch.randn((64,64)).numpy()
io.imsave('save.jpg', a)