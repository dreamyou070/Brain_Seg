import torch
from torch import nn
import torch.nn.functional as F


# [1] direct cross entropy
head_output = torch.randn((9,4)) # total class = 4
probability = torch.softmax(head_output, dim=1)
logwise_probability = torch.log(probability)
target = torch.randint(0,4,(9,))

criteria = torch.nn.CrossEntropyLoss()
loss = criteria(head_output, target.long())
print(f'loss = {loss}')
negative_perclass_log_likelihood = F.nll_loss(logwise_probability, target.long())
print(f'negative_perclass_log_likelihood = {negative_perclass_log_likelihood}')
# make one-hot target
target_onehot = F.one_hot(target, num_classes=4)
loss1 = (logwise_probability * target_onehot) * -1

print(f'loss1 = {loss1}')



def NLLLoss(model_output, targets):

    """ proposed personalized loss """

    p = torch.softmax(model_output, dim=1)
    log_p = torch.log(p)

    activating_out = torch.zeros_like(targets, dtype=torch.float)
    deactivating_out = torch.zeros_like(targets, dtype=torch.float)
    for i in range(len(targets)):
        act = log_p[i][targets[i]]
        deact = log_p[i].sum() - act
        activating_out[i] = act
        deactivating_out[i] = deact
    act_loss = -activating_out.sum()/len(activating_out)
    deact_loss = -deactivating_out.sum()/len(deactivating_out)
    return act_loss, deact_loss
act_loss, deact_loss= NLLLoss(head_output, target.long())
print(f'act_loss = {act_loss}')
print(f'deact_loss = {deact_loss}')