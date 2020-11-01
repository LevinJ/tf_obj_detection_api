from utils import get_dataset
import cv2
import matplotlib
matplotlib.use("TkAgg")

dataset = get_dataset("/home/levin/workspace/carnd/tf_obj_detection_api/data/processed/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord",
                      '../label_map.pbtxt')
def draw_detections(img, boxes, classes):
    color_dict = {1: (0, 0, 255), 2: (255, 0, 0), 4:(0, 255, 0)}
    for box,box_class in zip(boxes, classes):
        ymin, xmin, ymax, xmax = box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_dict[box_class], 2)
    return img

        
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

def display_instances(batch):
    """
    This function takes a batch from the dataset and display the image with 
    the associated bounding boxes.
    """
    # ADD CODE HERE
    return 
import matplotlib.pyplot as plt
fig=plt.figure()
rows, columns = 2, 2
i = 1
def display_dataset(dataset):
    dataset = dataset.shuffle(100)
    dataset = dataset.take(10)
    for elem in dataset:
        global i;
        img  = elem['image'].numpy()
        h, w, _c = img.shape 
        boxes = elem['groundtruth_boxes'].numpy()
        boxes[:,(0,2)] *= h
        boxes[:,(1,3)] *= w
        classes = elem['groundtruth_classes'].numpy()
        
        img = draw_detections(img.copy(), boxes, classes)
        print("show image{}".format(i))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        if i == rows * columns:
            break
        i = i+ 1
    plt.show()
#         print("show image")
#         cv2.imshow('object detection', cv2.resize(img, (0,0), fx=0.5, fy=0.5)  )
#         if cv2.waitKey(2500) & 0xFF == ord('q'):
#             continue
#     cv2.destroyAllWindows()
    return
    
display_dataset(dataset)   
