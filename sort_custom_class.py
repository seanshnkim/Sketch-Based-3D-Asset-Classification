import os 
import shutil
import pandas as pd

# For train data
source_fPath = '/home/ubuntu/Sketch-Recommendation/data/kaggle_ImageNet_data/ILSVRC/Data/CLS-LOC/train'
dest_fPath = '/home/ubuntu/Sketch-Recommendation/data/custom_ImageNet_data/train'

# custom class list
custom_cls_list_df = pd.read_csv('/home/ubuntu/Sketch-Recommendation/data/custom_class_list.csv')
custom_sys_IDs = custom_cls_list_df['synset_ID']

# idx = 0
# for (root, dirs, files) in os.walk(source_fPath):
#     curr_sys_id = dirs[idx]
#     if curr_sys_id in custom_sys_IDs:
#         new_path = os.path.join(dest_fPath, curr_sys_id)
#         if not os.path.exists(new_path):
#             os.makedirs(new_path)
#         shutil.copy(os.path.join(root, files), os.path.join(new_path, files))

imageNet_synset_IDs = os.listdir(source_fPath)

absent_synset_IDs = []
for cust_id in custom_sys_IDs:
    source_dir = os.path.join(source_fPath, cust_id)
    dest_dir = os.path.join(dest_fPath, cust_id)
    if os.path.exists(source_dir) and not os.path.exists(dest_dir):
        shutil.copytree(source_dir, dest_dir)
    else:
        absent_synset_IDs.append(cust_id)

pd.DataFrame(absent_synset_IDs).to_csv('/home/ubuntu/Sketch-Recommendation/results/absent_synset_IDs.csv', index=True)


