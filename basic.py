import os

test_folder = r'/home/dreamyou070/MyData/anomaly_detection/medical/leader_polyp/bkai-igh-neopolyp/test/anomal/mask_256'
train_folder = r'/home/dreamyou070/MyData/anomaly_detection/medical/leader_polyp/bkai-igh-neopolyp/train/anomal/mask_256'
test_list = os.listdir(test_folder)
train_list = os.listdir(train_folder)
for test_file in test_list :
    t = test_file.replace('.png', '')
    org_path = os.path.join(test_folder, test_file)
    new_path = os.path.join(test_folder,t)
    os.rename(org_path, new_path)
for train_file in train_list :
    t = train_file.replace('.png', '')
    org_path = os.path.join(train_folder, train_file)
    new_path = os.path.join(train_folder,t)
    os.rename(org_path, new_path)
