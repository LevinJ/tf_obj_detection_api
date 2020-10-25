from utils import get_dataset

dataset = get_dataset("/home/levin/workspace/carnd/tf_obj_detection_api/data/processed/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord",
                      '../label_map.pbtxt')

def display_instances(batch):
    """
    This function takes a batch from the dataset and display the image with 
    the associated bounding boxes.
    """
    # ADD CODE HERE
    
