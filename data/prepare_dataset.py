import os
from data.dataset import TrainDataset, TrainDataset_Unsupervised
import torch


def call_dataset(args) :

    # [1] set root data
    root_dir = os.path.join(args.data_path, f'train')

    tokenizer = None
    if not args.on_desktop :
        from model.tokenizer import load_tokenizer
        tokenizer = load_tokenizer(args)
    print(f'root_dir = {root_dir}')
    
    dataset_class = TrainDataset
    dataset = dataset_class(root_dir=root_dir,
                            resize_shape=[512, 512],
                            tokenizer=tokenizer,
                            caption=args.trigger_word,
                            latent_res=args.latent_res,)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    return dataloader

