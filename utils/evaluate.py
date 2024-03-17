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
def calculate_IOU(segmentation_model, dataloader, device, text_encoder, unet, vae, controller, weight_dtype,
             position_embedder, args):
    segmentation_model.eval()
    num_val_batches = len(dataloader)
    # iterate over the validation set
    with torch.no_grad():
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
            # [1] pred
            mask_pred = segmentation_model(q_dict[64], q_dict[32], q_dict[16])  # 1,4,64,64
            mask_pred = mask_pred.permute(0, 2, 3, 1).detach().cpu().numpy() # 1,64,64,4
            mask_pred_argmax = np.argmax(mask_pred, axis=3) # 1,64,64
            # [2] real (1,4,64,64)
            mask_true = mask_true.permute(0, 2, 3, 1).detach().cpu().numpy()
            # [3] IoU
            IOU_keras = MeanIoU(num_classes=4)
            IOU_keras.update_state(mask_pred_argmax, mask_true)
            break
    segmentation_model.train()

    values = np.array(IOU_keras.get_weights()).reshape(4,4)
    class0_IOU = values[0, 0] / (values[0, 0] + values[0, 1] + values[0, 2] + values[0, 3])
    class1_IOU = values[1, 1] / (values[1, 0] + values[1, 1] + values[1, 2] + values[1, 3])
    class2_IOU = values[2, 2] / (values[2, 0] + values[2, 1] + values[2, 2] + values[2, 3])
    class3_IOU = values[1, 1] / (values[1, 0] + values[1, 1] + values[1, 2] + values[1, 3])
    return IOU_keras, class0_IOU, class1_IOU, class2_IOU, class3_IOU, mask_pred_argmax.squeeze()