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

    if args.trigger_word == 'brain' :

        if not args.train_segmentation :
            dataset_class = TrainDataset_Single
            train_dataset = dataset_class(root_dir=root_dir,
                                        resize_shape=[512, 512],
                                        tokenizer=tokenizer,
                                        caption='brain',
                                        latent_res=args.latent_res, )
            test_dataset = train_dataset
        else :
            train_dataset = TrainDataset_Seg(root_dir=root_dir,
                                             resize_shape=[512, 512],
                                             tokenizer=tokenizer,
                                             caption='brain',
                                             latent_res=args.latent_res,
                                             n_classes = args.n_classes)
            test_dataset = TrainDataset_Seg(root_dir=os.path.join(args.data_path, f'test'),
                                             resize_shape=[512, 512],
                                             tokenizer=tokenizer,
                                             caption='brain',
                                             latent_res=args.latent_res,
                                             n_classes=args.n_classes)
    else :
        from data.dataset_teeth import TrainDataset


    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False)

    return train_dataloader, test_dataloader