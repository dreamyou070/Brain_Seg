import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

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


def reshape_batch_dim_to_heads(tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = 8
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len,
                            dim)  # 1,8,pix_num, dim -> 1,pix_nun, 8,dim
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len,
                                                dim * head_size)  # 1, pix_num, long_dim
    res = int(seq_len ** 0.5)
    tensor = tensor.view(batch_size // head_size, res, res, dim * head_size)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor

@torch.inference_mode()
def evaluation_check(segmentation_head, dataloader, device, text_encoder, unet, vae, controller, weight_dtype,
                  position_embedder, args):
    segmentation_head.eval()
    num_val_batches = len(dataloader)
    # iterate over the validation set
    with torch.no_grad():
        global_num = 0
        y_true_list, y_pred_list = [], []
        dice_coeff_list = []

        for global_num, batch in enumerate(dataloader):
            encoder_hidden_states = text_encoder(batch["input_ids"].to(device))["last_hidden_state"]
            if args.text_truncate :
                encoder_hidden_states = encoder_hidden_states[:,:2,:]
            # ------------------------------------------------------------------------------------------------------------
            image = batch['image'].to(dtype=weight_dtype)                                   # 1,3,512,512
            gt = batch['gt'].to(dtype=weight_dtype)                                         # 1,4,128,128
            gt_flat = batch['gt_flat'].to(dtype=weight_dtype)                               # 1,128*128
            with torch.no_grad():
                latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor
            with torch.set_grad_enabled(True):
                unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list, noise_type=position_embedder)
            query_dict, key_dict, attn_dict = controller.query_dict, controller.key_dict, controller.attn_dict
            controller.reset()
            q_dict = {}
            k_dict = {}

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = 8
                tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim) # 1,8,pix_num, dim -> 1,pix_nun, 8,dim
                tensor = tensor.permute(0, 2, 1, 3).contiguous().reshape(batch_size // head_size, seq_len, dim * head_size) # 1, pix_num, long_dim
                res = int(seq_len ** 0.5)
                tensor = tensor.view(batch_size // head_size, res,res, dim * head_size).contiguous()
                tensor = tensor.permute(0,3,1,2).contiguous()  # 1, dim, res,res
                return tensor

            for layer in args.trg_layer_list:
                query = query_dict[layer][0].squeeze()  # head, pix_num, dim
                res = int(query.shape[1] ** 0.5)
                q_dict[res] = reshape_batch_dim_to_heads(query) # 1, res,res,dim
                k_dict[res] = key_dict[layer][0]
            x16_out, x32_out, x64_out = q_dict[16], q_dict[32], q_dict[64]
            key64 = k_dict[64]
            # x16_out, x32_out, x64_out = [1,dim,res,res]
            if not args.cross_attn_base :
                masks_pred = segmentation_head(x16_out, x32_out, x64_out) # 1,4,128,128
            else :
                masks_pred = segmentation_head(x16_out, x32_out, x64_out, key64)  # 1,4,128,128
            #######################################################################################################################
            # segmentation model
            # [1] pred
            mask_pred_ = masks_pred.permute(0, 2, 3, 1).detach().cpu().numpy()  # 1,128,128,4
            mask_pred_argmax = np.argmax(mask_pred_, axis=3).flatten()          # 128*128
            y_pred_list.append(torch.Tensor(mask_pred_argmax))

            mask_true = gt_flat.detach().cpu().numpy().flatten()                # 128*128
            y_true_list.append(torch.Tensor(mask_true))

            # [2] dice coefficient
            from utils.dice_score import dice_loss
            dice_coeff = 1-dice_loss(F.softmax(masks_pred, dim=1).float(),  # class 0 ~ 4 check best
                                      gt,multiclass=True)
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