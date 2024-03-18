import os
import torch
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
                ct_mask_folder = os.path.join(ct_normality_folder, f'feature_{res}_mask')
                os.makedirs(ct_mask_folder, exist_ok=True)
                features = os.listdir(c1_res_folder)
                for feature in features :
                    c1_feature = torch.load(os.path.join(c1_res_folder, feature))
                    #c2_feature = torch.load(os.path.join(c2_res_folder, feature))
                    #c3_feature = torch.load(os.path.join(c3_res_folder, feature))










if __name__ == '__main__' :
    main()