import argparse, math, random, json
from tqdm import tqdm
from accelerate.utils import set_seed
import torch
from torch import nn
import os
from attention_store import AttentionStore
from data import call_dataset
from model import call_model_package
from model.segmentation_unet import (Segmentation_Head_a, Segmentation_Head_b, Segmentation_Head_c, \
    Segmentation_Head_a_with_binary, Segmentation_Head_b_with_binary, Segmentation_Head_c_with_binary)
from model.diffusion_model import transform_models_if_DDP
from model.unet import unet_passing_argument
from utils import prepare_dtype, arg_as_list, reshape_batch_dim_to_heads
from utils.attention_control import passing_argument, register_attention_control
from utils.accelerator_utils import prepare_accelerator
from utils.optimizer import get_optimizer, get_scheduler_fix
from utils.saving import save_model
from utils.loss import FocalLoss
from utils.evaluate import evaluation_check
from torch.nn import functional as F
from utils.loss import deactivating_loss
from ignite.engine import *
from ignite.metrics import *

def eval_step(engine, batch):
    return batch

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


def main(args):

    print(f'\n step 1. setting')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    args.logging_dir = os.path.join(output_dir, 'log')
    os.makedirs(args.logging_dir, exist_ok=True)
    record_save_dir = os.path.join(output_dir, 'record')
    os.makedirs(record_save_dir, exist_ok=True)
    with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(f'\n step 2. dataset and dataloader')
    if args.seed is None :
        args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)
    train_dataloader, test_dataloader = call_dataset(args)

    print(f'\n step 3. preparing accelerator')
    accelerator = prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    print(f'\n step 4. model')
    weight_dtype, save_dtype = prepare_dtype(args)
    text_encoder, vae, unet, network, position_embedder = call_model_package(args, weight_dtype, accelerator)

    if args.do_binary :
        model_class = Segmentation_Head_a_with_binary
        if args.aggregation_model_b:
            model_class = Segmentation_Head_b_with_binary
        if args.aggregation_model_c:
            model_class = Segmentation_Head_c_with_binary
    else :
        model_class = Segmentation_Head_a
        if args.aggregation_model_b :
            model_class = Segmentation_Head_b
        if args.aggregation_model_c :
            model_class = Segmentation_Head_c
    segmentation_head = model_class(n_classes=args.n_classes,
                                    mask_res=args.mask_res,
                                    norm_type=args.norm_type,
                                    use_nonlinearity=args.use_nonlinearity,
                                    nonlinear_type=args.nonlinearity_type,)

    print(f'\n step 5. optimizer')
    args.max_train_steps = len(train_dataloader) * args.max_train_epochs
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    if args.use_position_embedder:
        trainable_params.append({"params": position_embedder.parameters(), "lr": args.learning_rate})
    trainable_params.append({"params": segmentation_head.parameters(), "lr": args.learning_rate})
    optimizer_name, optimizer_args, optimizer = get_optimizer(args, trainable_params)

    print(f'\n step 6. lr')
    lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

    print(f'\n step 7. loss function')
    loss_CE = nn.CrossEntropyLoss()
    loss_FC = FocalLoss()
    loss_BCE = nn.BCELoss()

    print(f'\n step 8. model to device')
    if args.use_position_embedder :
        segmentation_head, unet, text_encoder, network, optimizer, train_dataloader, test_dataloader, lr_scheduler, position_embedder = \
            accelerator.prepare(segmentation_head, unet, text_encoder, network, optimizer, train_dataloader,
                                test_dataloader, lr_scheduler, position_embedder)
    else :
        segmentation_head, unet, text_encoder, network, optimizer, train_dataloader, test_dataloader, lr_scheduler = \
            accelerator.prepare(segmentation_head, unet, text_encoder, network, optimizer, train_dataloader,
                                test_dataloader, lr_scheduler)
    text_encoders = transform_models_if_DDP([text_encoder])
    unet, network = transform_models_if_DDP([unet, network])
    if args.use_position_embedder:
        position_embedder = transform_models_if_DDP([position_embedder])[0]
    if args.gradient_checkpointing:
        unet.train()
        position_embedder.train()
        segmentation_head.train()
        for t_enc in text_encoders:
            t_enc.train()
            if args.train_text_encoder:
                t_enc.text_model.embeddings.requires_grad_(True)
        if not args.train_text_encoder:  # train U-Net only
            unet.parameters().__next__().requires_grad_(True)
    else:
        unet.eval()
        for t_enc in text_encoders:
            t_enc.eval()
    del t_enc
    network.prepare_grad_etc(text_encoder, unet)
    vae.to(accelerator.device, dtype=weight_dtype)

    print(f'\n step 9. registering saving tensor')
    controller = AttentionStore()
    register_attention_control(unet, controller)

    print(f'\n step 10. Training !')
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0,
                        disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    loss_list = []

    for epoch in range(args.start_epoch, args.max_train_epochs):

        epoch_loss_total = 0
        accelerator.print(f"\nepoch {epoch + 1}/{args.start_epoch + args.max_train_epochs}")

        for step, batch in enumerate(train_dataloader):
            # [1] model forward
            device = accelerator.device
            loss_dict = {}
            with torch.set_grad_enabled(True):
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))["last_hidden_state"]
            image = batch['image'].to(dtype=weight_dtype)                                   # 1,3,512,512
            gt_flat = batch['gt_flat'].to(dtype=weight_dtype)                               # 1,128*128
            gt = batch['gt'].to(dtype=weight_dtype)                                         # 1,class_num,256,256
            with torch.no_grad():
                latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor
            with torch.set_grad_enabled(True):
                if args.use_position_embedder :
                    unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list, noise_type=position_embedder)
                else :
                    unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list)
            query_dict, key_dict = controller.query_dict, controller.key_dict
            controller.reset()
            q_dict = {}
            for layer in args.trg_layer_list:
                position = layer.split('_')[0]
                query = query_dict[layer][0].squeeze()  # head, pix_num, dim
                res = int(query.shape[1] ** 0.5)
                q_dict[f"{position}_{res}"] = reshape_batch_dim_to_heads(query) # 1, res,res,dim
            x16_out, x32_out, x64_out = q_dict['up_16'], q_dict['up_32'], q_dict['up_64']
            # [2] segmentation head
            if args.do_binary :
                binary_pred, masks_pred = segmentation_head(x16_out, x32_out, x64_out) # 1,4,128,128
            else :
                masks_pred = segmentation_head(x16_out, x32_out, x64_out)  # 1,4,128,128
            masks_pred_ = masks_pred.permute(0, 2, 3, 1).contiguous()              # 1,128,128,4
            masks_pred_ = masks_pred_.view(-1, masks_pred_.shape[-1]).contiguous() # pix_nuum, class_num

            # ------------------------------------------------------------------------------------------------------
            # [5.1] Multiclassification Loss
            loss = loss_CE(masks_pred_,gt_flat.squeeze().to(torch.long))  # 128*128
            loss_dict['multi_class_loss'] = loss.item()

            # [5.2] Focal Loss
            if args.do_focal_loss :
                loss_focal = loss_FC(masks_pred_, gt_flat.squeeze().to(torch.long))
                loss += loss_focal
                loss_dict['focal_loss'] = loss_focal.item()

            # [5.3] binary loss
            if args.do_binary:
                sigmoid = nn.Sigmoid()
                binary_pred_ = binary_pred.permute(0, 2, 3, 1).contiguous()  # 1,256,256,2
                binary_pred_ = binary_pred_.view(-1, binary_pred_.shape[-1]).contiguous()  # 256*256, 2
                # later try to change 0,1 class number
                binary_gt_flat = torch.where(gt_flat != 0, 1, 0).squeeze()  # 256*256
                binary_gt_flat = torch.nn.functional.one_hot(binary_gt_flat.to(torch.int64), num_classes=2)
                binary_loss = loss_BCE(sigmoid(binary_pred_), binary_gt_flat.to(weight_dtype))
                loss_dict['binary_loss'] = binary_loss.item()
                loss += binary_loss


            """
            # [5.4]
            if args.do_attn_loss:
                if batch['sample_idx'] == 1 :
                    gt = gt.permute(0, 2, 3, 1).contiguous()  # 1,256,256,4
                    gt = gt.view(-1, gt.shape[-1])              # 128*128,4
                    masks_pred_permute = masks_pred.permute(0, 2, 3, 1).contiguous()  # 1,res,res,4
                    masks_pred_permute = torch.softmax(masks_pred_permute, dim=-1)
                    masks_pred_permute = masks_pred_permute.view(-1, masks_pred_permute.shape[-1])  # 128*128,4
                    class_num = masks_pred_permute.shape[-1]
                    for class_idx in range(class_num):
                        if class_idx == 2 :
                            pred_attn_vector = masks_pred_permute[:, class_idx].squeeze()  # 128*128
                            activation_position = gt[:, class_idx]  # 128*128
                            deactivation_position = 1 - activation_position  # many 1
                            total_attn = torch.ones_like(pred_attn_vector)
                            activation_loss = (1 - ((pred_attn_vector * activation_position) / total_attn) ** 2).mean()
                            deactivation_loss = (((pred_attn_vector * deactivation_position) / total_attn) ** 2).mean()
                            loss += activation_loss + deactivation_loss
                            #loss += deactivation_loss
            if args.do_attn_loss_anomaly :
                gt = gt.permute(0, 2, 3, 1).contiguous()  # 1,256,256,4
                gt = gt.view(-1, gt.shape[-1])              # 128*128,4
                masks_pred_permute = masks_pred.permute(0, 2, 3, 1).contiguous()  # 1,res,res,4
                masks_pred_permute = torch.softmax(masks_pred_permute, dim=-1)
                masks_pred_permute = masks_pred_permute.view(-1, masks_pred_permute.shape[-1])  # 128*128,4
                class_num = masks_pred_permute.shape[-1]
                normal_position = gt[:,0]
                for class_idx in range(class_num):
                    if class_idx != 0 :
                        pred_attn_vector = masks_pred_permute[:, class_idx].squeeze()  # 128*128
                        activation_position = gt[:, class_idx]  # 128*128
                        deactivation_position = 1 - activation_position
                        total_attn = torch.ones_like(pred_attn_vector)
                        activation_loss = (1 - ((pred_attn_vector * activation_position) / total_attn) ** 2).mean()
                        deactivation_loss = (((pred_attn_vector * deactivation_position) / total_attn) ** 2).mean()
                        loss += activation_loss + deactivation_loss
            """
            # [5.2] back prop
            loss = loss.to(weight_dtype)
            current_loss = loss.detach().item()
            if epoch == args.start_epoch:
                loss_list.append(current_loss)
            else:
                epoch_loss_total -= loss_list[step]
                loss_list[step] = current_loss
            epoch_loss_total += current_loss
            avr_loss = epoch_loss_total / len(loss_list)
            loss_dict['avr_loss'] = avr_loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if is_main_process:
                progress_bar.set_postfix(**loss_dict)
            if global_step >= args.max_train_steps:
                break
        # ----------------------------------------------------------------------------------------------------------- #
        accelerator.wait_for_everyone()
        if is_main_process:
            saving_epoch = str(epoch+1).zfill(6)
            save_model(args,
                       saving_folder='model',
                       saving_name = f'lora-{saving_epoch}.safetensors',
                       unwrapped_nw=accelerator.unwrap_model(network),
                       save_dtype=save_dtype)
            if position_embedder is not None:
                save_model(args,
                           saving_folder='position_embedder',
                           saving_name = f'position_embedder-{saving_epoch}.safetensors',
                           unwrapped_nw=accelerator.unwrap_model(position_embedder),
                           save_dtype=save_dtype)
            save_model(args,
                       saving_folder='segmentation',
                       saving_name = f'segmentation-{saving_epoch}.safetensors',
                       unwrapped_nw=accelerator.unwrap_model(segmentation_head),
                       save_dtype=save_dtype)
        # ----------------------------------------------------------------------------------------------------------- #
        # [7] evaluate
        check_loader = test_dataloader
        if args.check_training :
            check_loader = train_dataloader
            print(f'test with training data')
        IOU_dict, confusion_matrix = evaluation_check(segmentation_head, check_loader, accelerator.device,
                                                      text_encoder, unet, vae, controller, weight_dtype, position_embedder, args)
        # saving
        if is_main_process:
            print(f'  - precision dictionary = {IOU_dict}')
            print(f'  - confusion_matrix = {confusion_matrix}')
            # numpy to list
            import numpy as np
            confusion_matrix = confusion_matrix.tolist()
            confusion_save_dir = os.path.join(args.output_dir, 'confusion.txt')
            with open(confusion_save_dir, 'a') as f:
                f.write(f' epoch = {epoch + 1} \n')
                for i in range(len(confusion_matrix)):
                    for j in range(len(confusion_matrix[i])):
                        f.write(' ' + str(confusion_matrix[i][j]) + ' ')
                    f.write('\n')
                f.write('\n')

            score_save_dir = os.path.join(args.output_dir, 'score.txt')
            with open(score_save_dir, 'a') as f:
                dices = []
                f.write(f' epoch = {epoch + 1} | ')
                for k in IOU_dict:
                    dice = float(IOU_dict[k])
                    f.write(f'class {k} = {dice} ')
                    dices.append(dice)
                dice_coeff = sum(dices) / len(dices)
                f.write(f'| dice_coeff = {dice_coeff}')
                f.write(f'\n')

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # step 1. setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')
    # step 2. dataset
    parser.add_argument("--resize_shape", type=int, default=512)
    parser.add_argument('--train_data_path', type=str, default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--test_data_path', type=str, default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bottle')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--trigger_word', type=str)
    parser.add_argument("--latent_res", type=int, default=64)
    # step 3. preparing accelerator
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--log_prefix", type=str, default=None)
    parser.add_argument("--lowram", action="store_true", )
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precision", )
    parser.add_argument("--position_embedding_layer", type=str)
    parser.add_argument("--d_dim", default=320, type=int)
    # step 4. model
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='facebook/diffusion-dalle')
    parser.add_argument("--clip_skip", type=int, default=None,
                        help="use output of nth layer from back of text encoder (n>=1)")
    parser.add_argument("--max_token_length", type=int, default=None, choices=[None, 150, 225],
                        help="max token length of text encoder (default for 75, 150 or 225) / text encoder", )
    parser.add_argument("--vae_scale_factor", type=float, default=0.18215)
    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network")
    parser.add_argument("--network_dim", type=int, default=64, help="network dimensions (depends on each network) ")
    parser.add_argument("--network_alpha", type=float, default=4, help="alpha for LoRA weight scaling, default 1 ", )
    parser.add_argument("--network_dropout", type=float, default=None, )
    parser.add_argument("--network_args", type=str, default=None, nargs="*", )
    parser.add_argument("--dim_from_weights", action="store_true", )
    parser.add_argument("--n_classes", default=4, type=int)
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--mask_res", type=int, default=128)
    parser.add_argument("--position_embedder_weights", type=str, default=None)
    parser.add_argument("--use_position_embedder", action='store_true')
    parser.add_argument("--aggregation_model_b", action='store_true')
    parser.add_argument("--aggregation_model_c", action='store_true')
    parser.add_argument("--nonlinearity_type", type=str, default="relu", choices=["relu", "gelu", "silu", "mish", "leaky_relu"], )
    parser.add_argument("--with_4_layers", action='store_true')
    parser.add_argument("--do_binary", action='store_true')
    parser.add_argument("--use_nonlinearity", action='store_true')
    parser.add_argument("--do_attn_loss", action='store_true')
    parser.add_argument("--pretrained_segmentation_model", type=str)
    # step 5. optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
              help="AdamW , AdamW8bit, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, Lion, SGDNesterov,"
                "SGDNesterov8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP,"
                             "DAdaptLion, DAdaptSGD, AdaFactor", )
    parser.add_argument("--use_8bit_adam", action="store_true", help="use 8bit AdamW optimizer(requires bitsandbytes)",)
    parser.add_argument("--use_lion_optimizer", action="store_true", help="use Lion optimizer (requires lion-pytorch)",)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm, 0 for no clipping")
    parser.add_argument("--optimizer_args", type=str, default=None, nargs="*",
                        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") ', )
    # [lr]
    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module")
    parser.add_argument("--lr_scheduler_args", type=str, default=None, nargs="*",
                        help='additional arguments for scheduler (like "T_max=100")')
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts", help="scheduler to use for lr")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler (default is 0)", )
    parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1,
                        help="Number of restarts for cosine scheduler with restarts / cosine with restarts", )
    parser.add_argument("--lr_scheduler_power", type=float, default=1,
                        help="Polynomial power for polynomial scheduler / polynomial", )
    parser.add_argument('--text_encoder_lr', type=float, default=1e-5)
    parser.add_argument('--unet_lr', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--train_unet', action='store_true')
    parser.add_argument('--train_text_encoder', action='store_true')
    # [training]
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--max_train_epochs", type=int, default=None, )
    parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing")
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    # [loss]
    parser.add_argument("--use_focal_loss", action='store_true')
    parser.add_argument("--cross_entropy_focal_loss_both", action='store_true')
    parser.add_argument("--check_training", action='store_true')
    parser.add_argument("--do_dice_loss", action='store_true')
    parser.add_argument("--do_penalty_loss", action='store_true')
    parser.add_argument("--norm_type", type = str)
    parser.add_argument("--saving_original_query", action='store_true')
    parser.add_argument("--do_cross_entropy_loss", action='store_true')
    parser.add_argument("--do_focal_loss", action='store_true')
    parser.add_argument("--do_binar", action='store_true')
    # [saving]
    parser.add_argument("--save_model_as", type=str, default="safetensors", choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors)", )
    args = parser.parse_args()
    unet_passing_argument(args)
    passing_argument(args)
    from data.dataset_multi import passing_mvtec_argument
    passing_mvtec_argument(args)
    main(args)
