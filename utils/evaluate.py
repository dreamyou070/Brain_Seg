import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import reshape_batch_dim_to_heads
from utils.loss import multiclass_dice_coeff, dice_loss
import torch
import numpy as np
from sklearn.metrics import confusion_matrix


@torch.inference_mode()
def evaluation_check(segmentation_head, dataloader, device, text_encoder, unet, vae, controller, weight_dtype,
                     position_embedder, args):
    segmentation_head.eval()

    with torch.no_grad():
        y_true_list, y_pred_list = [], []
        dice_coeff_list = []
        for global_num, batch in enumerate(dataloader):
            with torch.set_grad_enabled(True):
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))["last_hidden_state"]
            image = batch['image'].to(dtype=weight_dtype)                                   # 1,3,512,512
            gt_flat = batch['gt_flat'].to(dtype=weight_dtype)                               # 1,128*128
            gt = batch['gt'].to(dtype=weight_dtype)                                         # 1,4,128,128
            with torch.no_grad():
                latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor
            with torch.set_grad_enabled(True):
                unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list, noise_type=position_embedder)
            query_dict, key_dict, attn_dict = controller.query_dict, controller.key_dict, controller.attn_dict
            controller.reset()
            q_dict = {}
            for layer in args.trg_layer_list:
                query = query_dict[layer][0].squeeze()  # head, pix_num, dim
                res = int(query.shape[1] ** 0.5)
                q_dict[res] = reshape_batch_dim_to_heads(query) # 1, res,res,dim
            x16_out, x32_out, x64_out = q_dict[16], q_dict[32], q_dict[64]
            masks_pred = segmentation_head(x16_out, x32_out, x64_out) # 1,4,128,128
            print(f'masks_pred (1,3,256,256) = {masks_pred.shape}')
            #######################################################################################################################
            # [1] pred
            mask_pred_ = masks_pred.permute(0, 2, 3, 1).detach().cpu().numpy()  # 1,128,128,4
            mask_pred_argmax = np.argmax(mask_pred_, axis=3).flatten()          # 128*128
            y_pred_list.append(torch.Tensor(mask_pred_argmax))

            mask_true = gt_flat.detach().cpu().numpy().flatten()                # 128*128
            y_true_list.append(torch.Tensor(mask_true))
            # [2] dice coefficient
            dice_coeff = 1-dice_loss(F.softmax(masks_pred, dim=1).float(),  # class 0 ~ 4 check best
                                     gt,
                                     multiclass=True)
            dice_coeff_list.append(dice_coeff.detach().cpu())
        y_hat = torch.cat(y_pred_list)
        y = torch.cat(y_true_list)
        score = confusion_matrix(y, y_hat)
        actual_axis, pred_axis = score.shape
        IOU_dict = {}
        for actual_idx in range(actual_axis):
            total_actual_num = score[actual_idx]
            total_actual_num = sum(total_actual_num)
            precision = score[actual_idx, actual_idx] / total_actual_num
            IOU_dict[actual_idx] = precision
        dice_coeff = np.mean(np.array(dice_coeff_list))
    segmentation_head.train()
    return IOU_dict, mask_pred_argmax.squeeze(), dice_coeff