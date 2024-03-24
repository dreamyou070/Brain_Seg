import torch
import numpy as np
confusion_matrix = np.array([[1,2],
                             [3,1]])
confusion_matrix = confusion_matrix.tolist()
print(confusion_matrix)
with open('example.txt' , 'w') as f:
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            f.write(str(confusion_matrix[i][j]) + ' ')
        f.write('\n')

