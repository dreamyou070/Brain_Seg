import os
import torch


def call_dataset(args) :

    # [1] set root data
    root_dir = os.path.join(args.data_path, f'train')

    tokenizer = None
    if not args.on_desktop :
        from model.tokenizer import load_tokenizer
        tokenizer = load_tokenizer(args)

    if args.train_single :
        from data.dataset_single import TrainDataset_Single
        dataset_class = TrainDataset_Single
        train_dataset = dataset_class(root_dir=root_dir,
                                resize_shape=[512, 512],
                                tokenizer=tokenizer,
                                caption='brain',
                                latent_res=args.latent_res, )
        test_dataset = train_dataset

    elif args.train_class12 :
        root_dir = os.path.join(args.data_path, f'train')
        test_root_dir = os.path.join(args.data_path, f'val')
        from data.dataset_class12 import TrainDataset_Multi, TestDataset_Multi
        print(f' dataset class 12')
        train_dataset = TrainDataset_Multi(root_dir=root_dir,
                                           resize_shape=[args.resize_shape, args.resize_shape],
                                           tokenizer=tokenizer,
                                           caption=args.trigger_word,
                                           latent_res=args.latent_res,
                                           num_classes = args.n_classes)
        test_dataset = train_dataset

    else :
        root_dir = os.path.join(args.data_path, f'train')
        test_root_dir = os.path.join(args.data_path, f'val')
        from data.dataset_multi import TrainDataset_Multi, TestDataset_Multi
        train_dataset = TrainDataset_Multi(root_dir=root_dir,
                                           resize_shape=[args.resize_shape, args.resize_shape],
                                           tokenizer=tokenizer,
                                           caption=args.trigger_word,
                                           latent_res=args.latent_res,)
        test_dataset = TestDataset_Multi(root_dir=test_root_dir,
                                         resize_shape=[args.resize_shape, args.resize_shape],
                                         tokenizer=tokenizer,
                                         caption=args.trigger_word,
                                         latent_res=args.latent_res, )
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    return train_dataloader, test_dataloader