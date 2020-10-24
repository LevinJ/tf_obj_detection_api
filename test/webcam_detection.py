import os
import tarfile
import urllib.request
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np


class App(object):
    def __init__(self):
        return
    def create_dir(self):
        DATA_DIR = os.path.join(os.getcwd(), 'data')
        MODELS_DIR = os.path.join(DATA_DIR, 'models')
        for dir in [DATA_DIR, MODELS_DIR]:
            if not os.path.exists(dir):
                os.mkdir(dir)
        self.MODELS_DIR = MODELS_DIR
        return
    def download_model(self):
        MODELS_DIR = self.MODELS_DIR
        # Download and extract model
        MODEL_DATE = '20200711'
        MODEL_NAME = 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8'
        MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
        MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
        MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
        PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
        PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
        PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
        if not os.path.exists(PATH_TO_CKPT):
            print('Downloading model. This may take a while... ', end='')
            urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
            tar_file = tarfile.open(PATH_TO_MODEL_TAR)
            tar_file.extractall(MODELS_DIR)
            tar_file.close()
            os.remove(PATH_TO_MODEL_TAR)
            print('Done')
        
        # Download labels file
        LABEL_FILENAME = 'mscoco_label_map.pbtxt'
        LABELS_DOWNLOAD_BASE = \
            'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
        PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
        if not os.path.exists(PATH_TO_LABELS):
            print('Downloading label file... ', end='')
            urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
            print('Done')
        self.PATH_TO_CFG = PATH_TO_CFG
        self.PATH_TO_CKPT = PATH_TO_CKPT
        self.PATH_TO_LABELS = PATH_TO_LABELS
        return
    @tf.function
    def detect_fn(self, image):
        """Detect objects in image."""
        detection_model = self.detection_model
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
    
        return detections, prediction_dict, tf.reshape(shapes, [-1])
    def load_model(self):
        PATH_TO_CFG = self.PATH_TO_CFG
        PATH_TO_CKPT = self.PATH_TO_CKPT
        tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)
        
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()
        self.detection_model = detection_model
        return
    def load_label_map(self):
        PATH_TO_LABELS = self.PATH_TO_LABELS
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
        return
    def run_prediction(self):
        category_index = self.category_index
        cap = cv2.VideoCapture(0)
        while True:
            # Read frame from camera
            ret, image_np = cap.read()
        
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
        
            # Things to try:
            # Flip horizontally
            # image_np = np.fliplr(image_np).copy()
        
            # Convert image to grayscale
            # image_np = np.tile(
            #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)
            print("start predicton")
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections, predictions_dict, shapes = self.detect_fn(input_tensor)
        
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
        
            viz_utils.visualize_boxes_and_labels_on_image_array(
                  image_np_with_detections,
                  detections['detection_boxes'][0].numpy(),
                  (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                  detections['detection_scores'][0].numpy(),
                  category_index,
                  use_normalized_coordinates=True,
                  max_boxes_to_draw=200,
                  min_score_thresh=.30,
                  agnostic_mode=False)
        
            # Display output
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
        
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return
    def run(self):
        self.create_dir()
        self.download_model()
        self.load_model()
        self.load_label_map()
        self.run_prediction()
        
        return
    
if __name__ == "__main__":   
    obj= App()
    obj.run()