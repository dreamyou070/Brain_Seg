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
        dataset = dataset_class(root_dir=root_dir,
                                resize_shape=[512, 512],
                                tokenizer=tokenizer,
                                caption='brain',
                                latent_res=args.latent_res, )

    else :
        root_dir = os.path.join(args.data_path, f'train_')
        from data.dataset_multi import TrainDataset_Multi
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
        print(f'caption = {caption}')
        dataset = TrainDataset_Multi(root_dir=root_dir,
                                     resize_shape=[512, 512],
                                     tokenizer=tokenizer,
                                     caption=caption,
                                     latent_res=args.latent_res,)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    return dataloader

