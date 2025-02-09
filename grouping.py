import pandas as pd
import os
from tqdm import tqdm

directories_path = 'UrbanSound8K/audio/'
df = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')

for index_num,row in tqdm(df.iterrows()):
    data = str(row['slice_file_name'])
    file_name = os.path.join(os.path.abspath(directories_path),'fold'+str(row["fold"])+'/',data)
    final_class_labels = str(row['class'])
    new_directory_path = os.path.join(os.path.abspath(directories_path),final_class_labels)+'/'
    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)
    os.rename(file_name,os.path.join(new_directory_path,data))
