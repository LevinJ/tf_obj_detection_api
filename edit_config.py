import argparse
import glob

import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
import os
import shutil


def edit(train_dir, eval_dir, batch_size, checkpoint, label_map):
    """
    edit the config file and save it to pipeline_new.config
    args:
    - train_dir [str]: path to train directory
    - eval_dir [str]: path to val OR test directory 
    - batch_size [int]: batch size
    - checkpoint [str]: path to pretrained model
    - label_map [str]: path to labelmap file
    """
    train_dir = os.path.abspath(train_dir)
    eval_dir = os.path.abspath(eval_dir)
    checkpoint = os.path.abspath(checkpoint)
    label_map = os.path.abspath(label_map)
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig() 
    with tf.gfile.GFile("pipeline.config", "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)  
    
    training_files = glob.glob(train_dir + '/*.tfrecord')
    evaluation_files = glob.glob(eval_dir + '/*.tfrecord')

    pipeline_config.train_config.batch_size = batch_size
    pipeline_config.train_config.fine_tune_checkpoint = checkpoint
    pipeline_config.train_input_reader.label_map_path = label_map
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = training_files

    pipeline_config.eval_input_reader[0].label_map_path = label_map
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = evaluation_files

    config_text = text_format.MessageToString(pipeline_config)             
    with tf.gfile.Open("./pipeline_new.config", "wb") as f:                                                                                                                                                                                                                       
        f.write(config_text) 
        
    dst =  "./training/reference/pipeline_new.config"
    os.makedirs("./training/reference", exist_ok=True)
    shutil.copyfile("./pipeline_new.config", dst) 
    print("copy file {}".format(dst))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('--train_dir', required=False, type=str, default="./data/train/",
                        help='training directory')
    parser.add_argument('--eval_dir', required=False, type=str, default="./data/eval/",
                        help='validation or testing directory')
    parser.add_argument('--batch_size', required=False, type=int, default= 2,
                        help='number of images in batch')
    parser.add_argument('--checkpoint', required=False, type=str, default="./training/pretrained-models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0",
                        help='checkpoint path')  
#     parser.add_argument('--batch_size', required=False, type=int, default= 4,
#                         help='number of images in batch')
#     parser.add_argument('--checkpoint', required=False, type=str, default="./training/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0",
#                         help='checkpoint path')   
    parser.add_argument('--label_map', required=False, type=str, default="./label_map.pbtxt",
                        help='label map path')   
    args = parser.parse_args()
    edit(args.train_dir, args.eval_dir, args.batch_size, 
         args.checkpoint, args.label_map)
    