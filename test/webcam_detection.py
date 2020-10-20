import os
import tarfile
import urllib.request

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
        return
    def run(self):
        self.create_dir()
        self.download_model()
        return
    
if __name__ == "__main__":   
    obj= App()
    obj.run()