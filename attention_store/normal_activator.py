import torch
import torch.nn as nn


def passing_normalize_argument(args):
    global argument
    argument = args


class NormalActivator(nn.Module):

    def __init__(self, loss_focal, loss_l2, multiclassification_loss_fn,
                 use_focal_loss,
                 class_weight):
        super(NormalActivator, self).__init__()


        # [1]
        self.attention_loss_multi =  []
        self.attention_loss_dict = {}
        self.attention_loss_dict['normal_cls_loss'] = []
        self.attention_loss_dict['anomal_cls_loss'] = []
        self.attention_loss_dict['normal_trigger_loss'] = []
        self.attention_loss_dict['anomal_trigger_loss'] = []
        self.attention_loss_class_dict = {}
        self.trigger_score = []
        self.cls_score = []
        # [3]
        self.loss_focal = loss_focal
        self.loss_l2 = loss_l2
        self.multiclassification_loss_fn = multiclassification_loss_fn
        self.anomal_map_loss = []
        self.use_focal_loss = use_focal_loss
        # [4]
        self.normal_matching_query_loss = []
        self.resized_queries = []
        self.queries = []
        self.resized_attn_scores = []
        self.noise_prediction_loss = []
        self.resized_self_attn_scores = []
        self.class_weight = class_weight

    def collect_attention_scores_multi(self,
                                       attn_score, # [8, 64*64, 4]
                                       gt,         # [64*64, 4]
                                       do_normal_activating=True):
        attn_score = attn_score.squeeze()
        gt = gt.squeeze() # [res,res, c]

        seq_len = attn_score.shape[-1]
        for seq_idx in range(seq_len) :
            attn = attn_score[:, :, seq_idx].squeeze()     # [head,pix_num]
            head = attn.shape[0]
            attn_gt = gt[:, seq_idx].squeeze().flatten() # [pix_num]
            attn_gt = attn_gt.unsqueeze(0).repeat(head, 1) # [head, pix_num]
            total_score = torch.ones_like(attn_gt).to(attn.device)
            # [1] activating
            activating_loss = (1 - (attn * (attn_gt/total_score)) ** 2) # head, pix_num -> attention should be big
            # [2] deactivating
            deactivating_loss = (attn * ((1-attn_gt) / total_score)) ** 2
            if argument.do_class_weight :
                activating_loss = float(self.class_weight[seq_idx]) * activating_loss
                deactivating_loss = float(self.class_weight[seq_idx]) * deactivating_loss
            if seq_idx not in self.attention_loss_class_dict :
                self.attention_loss_class_dict[seq_idx] = []
            self.attention_loss_class_dict[seq_idx].append(activating_loss)
            self.attention_loss_class_dict[seq_idx].append(deactivating_loss)

    def collect_anomal_map_loss_multi_crossentropy(self,
                                                   attn_score, # [8, 64*64, 4]
                                                   gt_vector,         # [64*64]
                                                   do_normal_activating=True):
        attn_score = attn_score.squeeze()   # [8,res*res,4]
        attn_score = attn_score.mean(dim=0) # [res*res,4]
        gt_vector = gt_vector.squeeze().type(torch.LongTensor).to(attn_score.device) # [res*res]
        loss = self.multiclassification_loss_fn(attn_score, gt_vector) # what form ? (one value)
        self.anomal_map_loss.append(loss)

    def collect_attention_scores_single(self,
                                        attn_score,
                                        anomal_position_vector,
                                        do_normal_activating=True):
        # [1] preprocessing
        cls_score, trigger_score = attn_score.chunk(2, dim=-1)  # head, pix_num
        cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()  # head, pix_num
        cls_score, trigger_score = cls_score.mean(dim=0), trigger_score.mean(dim=0)  # pix_num
        total_score = torch.ones_like(cls_score)

        # [2]
        normal_cls_score = cls_score * (1-anomal_position_vector)           # ------------------------------------
        normal_trigger_score = trigger_score * (1-anomal_position_vector)
        anomal_cls_score = cls_score * anomal_position_vector
        anomal_trigger_score = trigger_score * anomal_position_vector

        # [3]
        normal_cls_score = normal_cls_score / total_score
        normal_trigger_score = normal_trigger_score / total_score
        anomal_cls_score = anomal_cls_score / total_score
        anomal_trigger_score = anomal_trigger_score / total_score

        # [4]
        normal_cls_loss = normal_cls_score ** 2
        normal_trigger_loss = (1 - normal_trigger_score ** 2)  # normal cls score 이랑 같은 상황
        anomal_cls_loss = (1 - anomal_cls_score ** 2)
        anomal_trigger_loss = anomal_trigger_score ** 2

        # [5]
        if do_normal_activating :
            self.attention_loss_dict['normal_cls_loss'].append(normal_cls_loss.mean())
            self.attention_loss_dict['normal_trigger_loss'].append(normal_trigger_loss.mean())

        anomal_pixel_num = anomal_position_vector.sum()
        if anomal_pixel_num > 0 : # if anomal sample ....
            self.attention_loss_dict['anomal_cls_loss'].append(anomal_cls_loss.mean())
            self.attention_loss_dict['anomal_trigger_loss'].append(anomal_trigger_loss.mean())

    def collect_anomal_map_loss_multi(self, attn_score, gt):

        attn_score = attn_score.squeeze() # [8,64*64,4]
        gt = gt.squeeze()                 # [64,64,4]
        seq_len = attn_score.shape[-1]
        for seq_idx in range(seq_len):
            attn = attn_score[:, :, seq_idx].squeeze()  # head, pix_num
            head = attn.shape[0]
            attn_gt = gt[:, :,seq_idx].squeeze().flatten()  # pix_num
            attn_gt = attn_gt.unsqueeze(0).repeat(head, 1)   # head, pix_num
            map_loss = self.loss_l2(attn.float(), attn_gt.float()).mean(dim=0)
            self.anomal_map_loss.append(map_loss)

    def collect_anomal_map_loss_single(self, attn_score, anomal_position_vector):

        cls_score, trigger_score = attn_score.chunk(2, dim=-1)  # [head,pixel], [head,pixel]
        cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()  # [head,pixel], [head,pixel]
        cls_score, trigger_score = cls_score.mean(dim=0), trigger_score.mean(dim=0)  # pix_num
        """ trigger score should be normal position """
        trg_trigger_score = 1 - anomal_position_vector
        map_loss = self.loss_l2(trigger_score.float(),
                                trg_trigger_score.float()) # be normal
        self.anomal_map_loss.append(map_loss)

    def generate_attention_loss_multi(self):

        class_ = self.attention_loss_class_dict.keys()
        activating_loss, deactivating_loss = [], []
        for class_idx in class_ :
            act_loss, deact_loss = self.attention_loss_class_dict[class_idx] # [head, pix_num]
            activating_loss.append(act_loss)
            deactivating_loss.append(deact_loss)
        activating_loss = torch.stack(activating_loss, dim=0).mean(dim=0)  # [num, head9, pix_num] -> [head, pix_num]
        deactivating_loss = torch.stack(deactivating_loss, dim=0).mean(dim=0)  # [num, head9, pix_num] -> [head, pix_num]
        self.attention_loss_class_dict = {}
        return activating_loss, deactivating_loss

    def generate_attention_loss_single(self):

        normal_cls_loss = torch.tensor(0.0, requires_grad=True)
        normal_trigger_loss = torch.tensor(0.0, requires_grad=True)
        if len(self.attention_loss_dict['normal_cls_loss']) != 0:
            normal_cls_loss = torch.stack(self.attention_loss_dict['normal_cls_loss'], dim=0).mean(dim=0)
            normal_trigger_loss = torch.stack(self.attention_loss_dict['normal_trigger_loss'], dim=0).mean(dim=0)

        anomal_cls_loss = torch.tensor(0.0, requires_grad=True)
        anomal_trigger_loss = torch.tensor(0.0, requires_grad=True)
        if len(self.attention_loss_dict['anomal_cls_loss']) != 0:
            anomal_cls_loss = torch.stack(self.attention_loss_dict['anomal_cls_loss'], dim=0).mean(dim=0)
            anomal_trigger_loss = torch.stack(self.attention_loss_dict['anomal_trigger_loss'], dim=0).mean(dim=0)

        self.attention_loss_dict = {'normal_cls_loss': [],
                                    'normal_trigger_loss': [],
                                    'anomal_cls_loss': [],
                                    'anomal_trigger_loss': []}
        return normal_cls_loss, normal_trigger_loss, anomal_cls_loss, anomal_trigger_loss


    def generate_anomal_map_loss(self):
        map_loss = torch.stack(self.anomal_map_loss, dim=0)
        map_loss = map_loss.mean()
        self.anomal_map_loss = []
        return map_loss

    def reset(self) -> None:

        # [1]
        self.attention_loss_multi = []
        self.attention_loss_dict = {}
        self.attention_loss_dict['normal_cls_loss'] = []
        self.attention_loss_dict['anomal_cls_loss'] = []
        self.attention_loss_dict['normal_trigger_loss'] = []
        self.attention_loss_dict['anomal_trigger_loss'] = []

        self.trigger_score = []
        self.cls_score = []
        self.anomal_map_loss = []
        # [4]
        self.normal_matching_query_loss = []
        self.resized_queries = []
        self.queries = []
        self.resized_attn_scores = []
        self.noise_prediction_loss = []
        self.resized_self_attn_scores = []