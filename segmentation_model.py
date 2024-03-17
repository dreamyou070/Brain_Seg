import argparse, math, random, json
from tqdm import tqdm
from accelerate.utils import set_seed
import torch
import os
from attention_store import AttentionStore
from attention_store.normal_activator import NormalActivator
from model.diffusion_model import transform_models_if_DDP
from model.unet import unet_passing_argument
from utils import get_epoch_ckpt_name, save_model, prepare_dtype, arg_as_list
from utils.attention_control import passing_argument, register_attention_control
from utils.accelerator_utils import prepare_accelerator
from utils.optimizer_utils import get_optimizer, get_scheduler_fix
from utils.model_utils import pe_model_save, te_model_save
from utils.utils_loss import FocalLoss
from data.prepare_dataset import call_dataset
from model import call_model_package
from attention_store.normal_activator import passing_normalize_argument
from torch import nn
from utils.dice_score import dice_loss
import torch.nn.functional as F
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
    train_dataloader = call_dataset(args)

    print(f'\n step 3. preparing accelerator')
    accelerator = prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    print(f'\n step 4. model ')
    weight_dtype, save_dtype = prepare_dtype(args)
    text_encoder, vae, unet, network, position_embedder = call_model_package(args, weight_dtype, accelerator)



    print(f'\n step 7. loss function')
    loss_focal = FocalLoss()
    loss_l2 = torch.nn.modules.loss.MSELoss(reduction='none')
    criterion = nn.CrossEntropyLoss()
    normal_activator = NormalActivator(loss_focal, loss_l2,
                                       multiclassification_loss_fn= criterion,
                                       use_focal_loss=args.use_focal_loss,
                                       class_weight=None)
    text_encoders = transform_models_if_DDP([text_encoder])
    unet, network = transform_models_if_DDP([unet, network])
    if args.use_position_embedder:
        position_embedder = transform_models_if_DDP([position_embedder])[0]
    if args.gradient_checkpointing:
        unet.train()
        position_embedder.train()
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
    network.to(accelerator.device, dtype=weight_dtype)
    position_embedder.to(accelerator.device, dtype=weight_dtype)

    print(f'\n step 9. registering saving tensor')
    controller = AttentionStore()
    register_attention_control(unet, controller)

    print(f'\n step 10. segmentation model')
    from model.segmentation_unet import UNet
    segmentation_model = UNet(n_classes=4, bilinear=False)
    args.max_train_steps = len(train_dataloader) * args.max_train_epochs
    trainable_params = []
    trainable_params.append({"params": segmentation_model.parameters(), "lr": args.learning_rate})
    optimizer_name, optimizer_args, optimizer = get_optimizer(args, trainable_params)
    lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)
    segmentation_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(segmentation_model, optimizer,
                                                                                        train_dataloader, lr_scheduler)

    print(f'\n step 10. Training !')
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0,
                        disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    loss_list = []

    for epoch in range(args.start_epoch, args.max_train_epochs):

        segmentation_model.train()
        epoch_loss_total = 0
        accelerator.print(f"\nepoch {epoch + 1}/{args.start_epoch + args.max_train_epochs}")

        for step, batch in enumerate(train_dataloader):
            device = accelerator.device
            loss_dict = {}
            with torch.no_grad():
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))["last_hidden_state"]
                if args.text_truncate :
                    encoder_hidden_states = encoder_hidden_states[:,:2,:]
                image = batch['image'].to(dtype=weight_dtype)                                   # 1,3,512,512
                true_masks = batch['gt'].to(dtype=weight_dtype).squeeze().flatten() # 1,4,64,64
                with torch.no_grad():
                    latents = vae.encode(image).latent_dist.sample() * args.vae_scale_factor
                with torch.set_grad_enabled(True):
                    unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list, noise_type=position_embedder)
                query_dict, key_dict, attn_dict = controller.query_dict, controller.key_dict, controller.attn_dict
                controller.reset()
                q_dict = {}
                for layer in args.trg_layer_list:
                    query = query_dict[layer][0].squeeze()  # head, pix_num, dim
                    head, pix_num, dim = query.shape
                    res = int(pix_num ** 0.5)
                    query = query.view(head, res, res, dim).permute(0,3,1,2).mean(dim=0)
                    q_dict[res] = query
            #######################################################################################################################
            # segmentation model
            masks_pred = segmentation_model(q_dict[64], q_dict[32], q_dict[16])
            # target = true mask

            with torch.autocast():
                loss = criterion(masks_pred, true_masks)
                loss_dict['cross_entropy_loss'] = loss.item()
                loss += dice_loss(F.softmax(masks_pred, dim=1).float(),
                                  F.one_hot(true_masks, segmentation_model.n_classes).permute(0, 3, 1, 2).float(),
                                  multiclass=True)
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
            normal_activator.reset()
            if global_step >= args.max_train_steps:
                break
        # ----------------------------------------------------------------------------------------------------------- #
        # [6] epoch final
        accelerator.wait_for_everyone()
        if is_main_process:
            segmentation_base_save_dir = os.path.join(args.output_dir, 'segmentation')
            os.makedirs(segmentation_base_save_dir, exist_ok=True)
            p_save_dir = os.path.join(segmentation_base_save_dir,
                                      f'segmentation_{epoch + 1}.safetensors')
            pe_model_save(accelerator.unwrap_model(segmentation_model), save_dtype, p_save_dir)
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # step 1. setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')

    # step 2. dataset
    parser.add_argument('--data_path', type=str, default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
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

    # step 5. optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
                  help="AdamW , AdamW8bit, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, Lion, SGDNesterov,"
                "SGDNesterov8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP,"
                             "DAdaptLion, DAdaptSGD, AdaFactor", )
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="use 8bit AdamW optimizer(requires bitsandbytes)", )
    parser.add_argument("--use_lion_optimizer", action="store_true",
                        help="use Lion optimizer (requires lion-pytorch)", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm, 0 for no clipping")
    parser.add_argument("--optimizer_args", type=str, default=None, nargs="*",
                        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") ', )
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

    # step 10. training
    parser.add_argument("--save_model_as", type=str, default="safetensors",
               choices=[None, "ckpt", "pt", "safetensors"], help="format to save the model (default is .safetensors)",)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--max_train_epochs", type=int, default=None, )
    parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing")
    parser.add_argument("--do_attn_loss", action='store_true')
    parser.add_argument("--do_cls_train", action='store_true')
    parser.add_argument("--attn_loss_weight", type=float, default=1.0)

    parser.add_argument("--activating_loss_weight", type=float, default=1.0)
    parser.add_argument('--deactivating_loss_weight', type=float, default=1.0)

    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--original_normalized_score", action='store_true')
    parser.add_argument("--do_map_loss", action='store_true')
    parser.add_argument("--use_focal_loss", action='store_true')
    parser.add_argument("--test_noise_predicting_task_loss", action='store_true')
    parser.add_argument("--dist_loss_with_max", action='store_true')
    parser.add_argument("--on_desktop", action='store_true')
    parser.add_argument("--all_positional_embedder", action='store_true')
    parser.add_argument("--use_position_embedder", action='store_true')
    parser.add_argument("--position_embedder_weights", type=str, default=None)
    parser.add_argument("--vae_pretrained_dir", type=str)
    parser.add_argument("--local_hidden_states_globalize", action='store_true')
    parser.add_argument("--normal_activating_test", action='store_true')
    parser.add_argument("--train_single", action='store_true')
    parser.add_argument("--resize_shape", type=int, default=512)
    parser.add_argument("--multiclassification_focal_loss", action='store_true')
    parser.add_argument("--do_class_weight", action='store_true')
    parser.add_argument("--text_truncate", action='store_true')
    args = parser.parse_args()
    unet_passing_argument(args)
    passing_argument(args)
    passing_normalize_argument(args)
    from data.dataset_multi import passing_mvtec_argument
    passing_mvtec_argument(args)
    main(args)