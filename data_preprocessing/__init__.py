import os
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
import numpy as np
def main() :

    print(f' step 1. data dir')
    base_dir = r'D:\medical\abdominal\data/chaos'
    phases = os.listdir(base_dir) # train, test
    for phase in phases :
        if 'test' not in phase.lower() :

            phase_dir = os.path.join(base_dir, f'{phase}')
            folders = os.listdir(phase_dir)
            for folder in folders :
                if folder == 'MR_pil'  :
                    folder_dir = os.path.join(phase_dir, folder)
                    pil_save_folder = os.path.join(phase_dir, f'{folder}_multimodal')
                    os.makedirs(pil_save_folder, exist_ok = True)
                    patient_files = os.listdir(folder_dir)
                    for patient_file in patient_files :
                        save_patient_dir = os.path.join(pil_save_folder, patient_file)
                        os.makedirs(save_patient_dir, exist_ok = True)
                        patient_file_folder = os.path.join(folder_dir, patient_file)
                        sub_folders = os.listdir(patient_file_folder) # T1DUAL, T2SPIR
                        for sub_folder in sub_folders :
                            if sub_folder == 'T1DUAL' :
                                inphase_dir = os.path.join(patient_file_folder, f'{sub_folder}/InPhase')
                                outphase_dir =os.path.join(patient_file_folder, f'{sub_folder}/OutPhase')
                                gt_dir = os.path.join(patient_file_folder, f'{sub_folder}/Ground')
                                save_dir = os.path.join(patient_file_folder, f'{sub_folder}/multimodal')
                                os.makedirs(save_dir, exist_ok = True)
                                images = os.listdir(inphase_dir)
                                for img in images :
                                    name, ext= os.path.splitext(img)
                                    name_list = name.split('-')
                                    final_num = str(int(name_list[2]) - 1).zfill(5)
                                    in_img_dir = os.path.join(inphase_dir, img)
                                    out_img_dir = os.path.join(outphase_dir, f'{name_list[0]}-{name_list[1]}-{final_num}.jpg')
                                    in_arr = np.array(Image.open(in_img_dir))
                                    out_arr = np.array(Image.open(out_img_dir))
                                    h,w = in_arr.shape
                                    back_arr = np.zeros((h,w,3))
                                    back_arr[:, :, 0] = in_arr
                                    back_arr[:, :, 1] = out_arr
                                    back_arr[:, :, 2] = in_arr
                                    pil = Image.fromarray(back_arr.astype(np.uint8))
                                    pil.save(os.path.join(save_dir,f'{name}.png'))





if __name__ == "__main__" :
    main()