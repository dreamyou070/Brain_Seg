import os
from data.dataset import TrainDataset
import torch


def call_dataset(args) :

    # [1] set root data
    root_dir = os.path.join(args.data_path, f'train')

    tokenizer = None
    if not args.on_desktop :
        from model.tokenizer import load_tokenizer
        tokenizer = load_tokenizer(args)
    print(f'root_dir = {root_dir}')

    if args.binary_test :
        from data.dataset_binary import TrainDataset_Binary
        dataset_class = TrainDataset_Binary
        dataset = dataset_class(root_dir=root_dir,
                                resize_shape=[512, 512],
                                tokenizer=tokenizer,
                                caption='brain',
                                latent_res=args.latent_res, )

    else :
        nectoric_word = ['necrotic']
        ederma_word = ['ederma']
        tumor_word = ['tumor', 'enhancing tumor']
        trigger_word = args.trigger_word.split(',')
        caption = ''
        for c in trigger_word:
            c = c.strip()
            if c in nectoric_word:
                target = 'n '
            elif c in ederma_word:
                target = 'e '
            else:
                target = 't '
            caption += target
        caption = caption.strip()
        # -------------------------------------------------------------------------------------------
        dataset_class = TrainDataset
        dataset = dataset_class(root_dir=root_dir,
                                resize_shape=[512, 512],
                                tokenizer=tokenizer,
                                caption=caption,
                                latent_res=args.latent_res,)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    return dataloader

