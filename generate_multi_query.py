import os
import torch
import numpy as np


def expand_mask(mask, head, dim):
    mask = np.expand_dims(mask, axis=0)
    mask = np.repeat(mask, repeats=head, axis=0, )
    mask = np.expand_dims(mask, axis=3)
    mask = np.repeat(mask, repeats=dim, axis=3, )
    mask = torch.tensor(mask)
    return mask

def main() :

    res_list = [16,32,64]

    print(f' step 1. ')
    base_folder = r'/home/dreamyou070/MyData/anomaly_detection/medical/brain'
    class_1_folder = os.path.join(base_folder, 'BraTS2020_Segmentation_class_1')
    class_2_folder = os.path.join(base_folder, 'BraTS2020_Segmentation_class_2')
    class_3_folder = os.path.join(base_folder, 'BraTS2020_Segmentation_class_3')
    class_t_folder = os.path.join(base_folder, 'BraTS2020_Segmentation_class_total')
    phases = os.listdir(class_1_folder)
    for phase in phases :
        c1_phase = os.path.join(class_1_folder, phase)
        c2_phase = os.path.join(class_2_folder, phase)
        c3_phase = os.path.join(class_3_folder, phase)
        ct_phase = os.path.join(class_t_folder, phase)
        os.makedirs(ct_phase, exist_ok = True)
        normality_folders = os.listdir(c1_phase)

        for normality_folder in normality_folders :
            c1_normality_folder = os.path.join(c1_phase, normality_folder)
            c2_normality_folder = os.path.join(c2_phase, normality_folder)
            c3_normality_folder = os.path.join(c3_phase, normality_folder)
            ct_normality_folder = os.path.join(ct_phase, normality_folder)
            os.makedirs(ct_normality_folder, exist_ok = True)

            for res in res_list :
                c1_res_folder = os.path.join(c1_normality_folder, f'feature_{res}')
                c2_res_folder = os.path.join(c2_normality_folder, f'feature_{res}')
                c3_res_folder = os.path.join(c3_normality_folder, f'feature_{res}')
                ct_res_folder = os.path.join(ct_normality_folder, f'feature_{res}')
                os.makedirs(ct_res_folder, exist_ok = True)
                if normality_folder == 'anormal' :
                    ct_mask_folder = os.path.join(ct_normality_folder, f'feature_{res}_mask')
                    os.makedirs(ct_mask_folder, exist_ok=True)
                features = os.listdir(c1_res_folder)
                for feature in features :
                    name, ext = os.path.splitext(feature)
                    c1_feature = torch.load(os.path.join(c1_res_folder, feature)) # head, dim, res, res
                    head, dim = c1_feature.shape[0], c1_feature.shape[1]
                    c1_feature = c1_feature.permute(0,2,3,1) # head, res,res,dim
                    c2_feature = torch.load(os.path.join(c2_res_folder, feature))
                    c2_feature = c2_feature.permute(0, 2, 3, 1)
                    c3_feature = torch.load(os.path.join(c3_res_folder, feature))
                    c3_feature = c3_feature.permute(0, 2, 3, 1)

                    if normality_folder == 'anormal' :
                        mask_arr = os.path.join(ct_mask_folder, f'{name}.npy')
                        mask = np.load(mask_arr) # [res,res]
                        c0_mask = np.where(mask == 0, 1, 0)
                        c1_mask = np.where(mask == 1, 1, 0)
                        c2_mask = np.where(mask == 2, 1, 0)
                        c3_mask = np.where(mask == 3, 1, 0)
                        c0_mask = expand_mask(c0_mask, head, dim)
                        c1_mask = expand_mask(c1_mask, head, dim)
                        c2_mask = expand_mask(c2_mask, head, dim)
                        c3_mask = expand_mask(c3_mask, head, dim)

                        # [4] total feature
                        total_feature = c1_feature * c1_mask + c2_feature * c2_mask + c3_feature * c3_mask + (c1_feature+c2_feature+c3_feature)/3 * c0_mask
                        total_feature = total_feature.permute(0,3,1,2)
                        torch.save(total_feature, os.path.join(ct_res_folder, feature))
                    else :
                        total_feature = (c1_feature + c2_feature + c3_feature)/3
                        total_feature = total_feature.permute(0, 3, 1, 2)
                        torch.save(total_feature, os.path.join(ct_res_folder, feature))













if __name__ == '__main__' :
    main()