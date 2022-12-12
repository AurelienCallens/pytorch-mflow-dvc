import os
import yaml
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('src/')
from utils.mlflow_run_decorator import mlflow_run


@mlflow_run
def split_dataset(annotations_file, output_folder, test_split=0.2, val_split=0.2, random_seed=99):
    dataset = pd.read_csv(annotations_file)
    train, test = train_test_split(dataset, stratify=dataset['label'], test_size=test_split, random_state=random_seed)
    
    train, val = train_test_split(train, stratify=train['label'], test_size=val_split, random_state=random_seed)

    print('Train set shape: {train_shape} \n Val set shape: {val_shape} \n Test set shape: {test_shape}'.format(
        train_shape=train.shape, val_shape=val.shape, test_shape=test.shape))
    
    mlflow.log_param("train_shape", train.shape)
    mlflow.log_param("val_shape", val.shape)
    mlflow.log_param("test_shape", test.shape)

    train.to_csv(os.path.join(output_folder, 'train.csv'), index=False)
    val.to_csv(os.path.join(output_folder, 'val.csv'), index=False)
    test.to_csv(os.path.join(output_folder, 'test.csv'), index=False)

if __name__ == '__main__':
    
    # Read params
    params = yaml.safe_load(open('./params.yaml'))['prepare']
    
    # Split dataset
    split_dataset(
        annotations_file='./data/interim/filepath.csv',
        output_folder='./data/processed',
        test_split=params['test_split'],
        val_split=params['val_split'],
        random_seed=99)