import argparse
import glob
import os
import random

import numpy as np
import shutil

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # TODO: Implement function
    training_files = glob.glob(data_dir + '/*.tfrecord')
    training_files.sort()
    num = len(training_files)
    
    # create the directry
    for _dir in ["train", "val", "test"]:
        dir_path = "{}/{}".format(data_dir, _dir)
        dir_path = os.path.abspath(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    
    #split out the training part, 0.8
    start = 0
    end = start + int(0.8 * num)
    for file_path in training_files[start:end]:
        dst_path = "{}/train/{}".format(data_dir, os.path.basename(file_path))
        shutil.move(file_path, dst_path)
    #split out the val part, 0.1
    start = end
    end = start + int(0.1 * num)
    for file_path in training_files[start:end]:
        dst_path = "{}/val/{}".format(data_dir, os.path.basename(file_path))
        shutil.move(file_path, dst_path)
    
    #split out the val part, 0.1
    for file_path in training_files[end:]:
        dst_path = "{}/test/{}".format(data_dir, os.path.basename(file_path))
        shutil.move(file_path, dst_path)
    return
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)