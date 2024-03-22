import os
import torch
from data.dataset_single import TrainDataset_Single
from data.dataset_multi import TrainDataset_Seg

def call_dataset(args) :

    # [1] set root data
    root_dir = os.path.join(args.data_path, f'train')

    tokenizer = None
    if not args.on_desktop :
        from model.tokenizer import load_tokenizer
        tokenizer = load_tokenizer(args)

    if args.trigger_word != 'teeth' :

        if not args.train_segmentation :
            dataset_class = TrainDataset_Single
            train_dataset = dataset_class(root_dir=root_dir,
                                        resize_shape=[512, 512],
                                        tokenizer=tokenizer,
                                        caption=args.trigger_word,
                                        latent_res=args.latent_res,
                                        mask_res = args.mask_res,)
            test_dataset = dataset_class(root_dir=os.path.join(args.data_path, f'test'),
                                        resize_shape=[512, 512],
                                        tokenizer=tokenizer,
                                        caption=args.trigger_word,
                                        latent_res=args.latent_res,
                                        mask_res = args.mask_res,)
        else :
            train_dataset = TrainDataset_Seg(root_dir=root_dir,
                                             resize_shape=[512, 512],
                                             tokenizer=tokenizer,
                                             caption=args.trigger_word,
                                             latent_res=args.latent_res,
                                             n_classes = args.n_classes,
                                             single_modality = args.single_modality,
                                             mask_res = args.mask_res,)
            test_dataset = TrainDataset_Seg(root_dir=os.path.join(args.data_path, f'test'),
                                             resize_shape=[512, 512],
                                             tokenizer=tokenizer,
                                             caption=args.trigger_word,
                                             latent_res=args.latent_res,
                                             n_classes=args.n_classes,
                                            single_modality = args.single_modality,
                                             mask_res = args.mask_res,)
    else :
        from data.dataset_teeth import TrainDataset_Seg as teeth_dataset
        train_dataset = teeth_dataset(root_dir=root_dir,
                                         resize_shape=[512, 512],
                                         tokenizer=tokenizer,
                                         caption=args.trigger_word,
                                         latent_res=args.latent_res,
                                         n_classes=args.n_classes)
        test_dataset = teeth_dataset(root_dir=os.path.join(args.data_path, f'test'),
                                        resize_shape=[512, 512],
                                        tokenizer=tokenizer,
                                        caption=args.trigger_word,
                                        latent_res=args.latent_res,
                                        n_classes=args.n_classes)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False)

    return train_dataloader, test_dataloader