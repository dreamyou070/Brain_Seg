import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff
import torch
import numpy as np
from keras.metrics import MeanIoU

@torch.inference_mode()
def evaluate(segmentation_model, dataloader, device, text_encoder, unet, vae, controller, weight_dtype, position_embedder, args):
    segmentation_model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    # iterate over the validation set
    with torch.no_grad() :
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch',
                          leave=False):
            encoder_hidden_states = text_encoder(batch["input_ids"].to(device))["last_hidden_state"]
            if args.text_truncate:
                encoder_hidden_states = encoder_hidden_states[:, :2, :]
            image, mask_true = batch['image'].to(dtype=weight_dtype), batch['gt'].to(dtype=weight_dtype)
            latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor
            unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
                 noise_type=position_embedder)
            query_dict, key_dict, attn_dict = controller.query_dict, controller.key_dict, controller.attn_dict
            controller.reset()
            q_dict = {}
            for layer in args.trg_layer_list:
                query = query_dict[layer][0].squeeze()  # head, pix_num, dim
                head, pix_num, dim = query.shape
                res = int(pix_num ** 0.5)
                query = query.view(head, res, res, dim).permute(0, 3, 1, 2).mean(dim=0)
                q_dict[res] = query.unsqueeze(0)
            #######################################################################################################################
            # segmentation model
            mask_pred = segmentation_model(q_dict[64], q_dict[32], q_dict[16])  # 1,4,64,64
            # target = true mask
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:],
                                                    reduce_batch_first=False)
            break
    segmentation_model.train()
    return dice_score



@torch.inference_mode()
def evaluation_check(segmentation_model, dataloader, device, text_encoder, unet, vae, controller, weight_dtype,
                  position_embedder, args):
    segmentation_model.eval()
    num_val_batches = len(dataloader)
    # iterate over the validation set
    with torch.no_grad():
        global_num = 0
        y_true_list, y_pred_list = [], []
        dice_coeff_list = []

        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch',
                          leave=False):
            if global_num < 100:
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))["last_hidden_state"]
                if args.text_truncate:
                    encoder_hidden_states = encoder_hidden_states[:, :2, :]
                image = batch['image'].to(dtype=weight_dtype)  # 1,3,512,512
                true_mask_one_hot_matrix = batch['gt'].to(dtype=weight_dtype)  # 1,4,64,64
                true_mask_one_vector = batch['gt_vector'].to(dtype=weight_dtype)  # 4096
                with torch.no_grad():
                    latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor
                with torch.set_grad_enabled(True):
                    unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
                         noise_type=position_embedder)
                query_dict, key_dict, attn_dict = controller.query_dict, controller.key_dict, controller.attn_dict
                controller.reset()
                q_dict = {}
                for layer in args.trg_layer_list:
                    query = query_dict[layer][0].squeeze()  # head, pix_num, dim
                    head, pix_num, dim = query.shape
                    res = int(pix_num ** 0.5)
                    query = query.view(head, res, res, dim).permute(0, 3, 1, 2).mean(dim=0)
                    q_dict[res] = query.unsqueeze(0)
                mask_pred = segmentation_model(q_dict[64], q_dict[32], q_dict[16])  # 1,4,64,64
                #######################################################################################################################
                # segmentation model
                # [1] pred
                mask_pred = mask_pred.permute(0, 2, 3, 1).detach().cpu().numpy()  # 1,64,64,4
                mask_pred_argmax = np.argmax(mask_pred, axis=3).flatten()
                y_pred_list.append(mask_pred_argmax)
                mask_true = true_mask_one_vector.detach().cpu().numpy().flatten()
                y_true_list.append(mask_true)

                # [2] dice coefficient
                dice_coeff = 1-dice_loss(F.softmax(masks_pred, dim=1).float(),  # class 0 ~ 4 check best
                                          true_mask_one_hot_matrix,  # true_masks = [1,4,64,64] (one-hot_
                                          multiclass=True)
                dice_coeff_list.append(dice_coeff)
                global_num += 1

        y = torch.cat(y_true_list)
        y_hat = torch.cat(y_pred_list)
        from sklearn.metrics import confusion_matrix
        score = confusion_matrix(y, y_hat)
        actual_axis, pred_axis = score.shape
        IOU_dict = {}
        for actual_idx in range(actual_axis):
            total_actual_num = score[actual_idx]
            total_actual_num = sum(total_actual_num)
            precision = score[actual_idx, actual_idx] / total_actual_num
            IOU_dict[actual_idx] = precision
        dice_coeff = np.mean(np.array(dice_coeff_list))
    segmentation_model.train()
    return IOU_dict, mask_pred_argmax.squeeze(), dice_coeff
