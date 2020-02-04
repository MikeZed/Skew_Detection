

from Model import Model

import tensorflow as tf
from tensorflow import keras
import os

YES = ['y', 'Y', '1', ' ']
MODELS_DIR = 'Models Objects'
# GRAPHS_DIR = 'Training Graphs'
MODEL_FILE = 'model'
MODEL_NUM = 0

## Model Parameters ##
# -- training --
EPOCHS = 150
BATCH_SIZE = 64
# -- structure and functions --
# NUM_OF_FILTERS = (32, 64, 256)
LEARNING_RATE = 0.001
OPTIMIZER = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
LOSS_FUNC = 'mean_squared_error'
METRICS = ['mean_absolute_error']

if LOSS_FUNC in METRICS:
    METRICS.remove(LOSS_FUNC)

MODEL_STRUCT = [

    {'name': 'In'},

    {'name': 'Conv2D', 'filters': 32, 'kernel_size': (5, 5)},
    {'name': 'BN'},
    {'name': 'Conv2D', 'filters': 32, 'kernel_size': (5, 5)},
    {'name': 'BN'},
    {'name': 'Conv2D', 'filters': 32, 'kernel_size': (5, 5)},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},

    {'name': 'MaxPooling2D', 'size': (5, 5)},

    {'name': 'Lambda', 'func': (lambda v: tf.abs(tf.signal.rfft2d(v)))},

    {'name': 'Conv2D', 'filters': 64, 'kernel_size': (3, 3)},
    {'name': 'BN'},
    {'name': 'Conv2D', 'filters': 64, 'kernel_size': (3, 3)},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},

    {'name': 'MaxPooling2D', 'size': (3, 3)},

    {'name': 'Conv2D', 'filters': 128, 'kernel_size': (3, 3)},
    {'name': 'Activation', 'type': 'relu'},

    {'name': 'AveragePooling2D', 'size': (2, 2)},

    {'name': 'Flatten'},

    {'name': 'Dense', 'size': 64},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},

    {'name': 'Dense', 'size': 1},
]
TRANSFER_MODEL = [

    {'name': 'VGG16'},
    {'name': 'Flatten'},
    {'name': 'Dense', 'size' : 1}

]
USE_TRANSFER = False
# -----------
MODEL_DICT = {'optimizer': OPTIMIZER, 'loss': LOSS_FUNC, 'metrics': METRICS,
              'struct': MODEL_STRUCT if not USE_TRANSFER else TRANSFER_MODEL}

TRAINING_DICT = {'epochs': EPOCHS, 'batch_size': BATCH_SIZE}
# FOLDERS_DICT = {'models': MODELS_DIR, 'graphs': GRAPHS_DIR}

## Image Settings ##
IMAGE_RES = 50  # setting this parameter for more the 200 is not recommended
IMG_MODE = 'pad'  # 'pad', 'patch' or 'edges'


def create_model(URL, DATA_PATH, DATA_FILE, split=(80, 10, 10),
                 use_generator=False, save_model=True):
    # checks if there is an existing model,
    # can create a new one by downloading and loading the data, creating a new model, training it and then
    # plots the results and saves them

    # ---------------------------------------
    #            check existing model
    # ---------------------------------------

    #print(MODEL_DICT)

    path = "{}\\{}".format(MODELS_DIR, MODEL_FILE)
    path += "" if MODEL_NUM == 0 else "_Num{}".format(MODEL_NUM)

    use_existing = prep_and_check_existing(path)

    # -------------------------------------------------
    #     load existing model or create a new one
    # -------------------------------------------------

    if not use_existing:
        # build and train new model

        if not save_model:
            path = None

        model = Model(use_generator=use_generator, img_channels=1, img_res=IMAGE_RES, img_mode=IMG_MODE)

        data_loader_dict = {'url': URL, 'data_path': DATA_PATH, 'data_file': DATA_FILE}

        model.construct(**MODEL_DICT, **data_loader_dict, **TRAINING_DICT, split=split, save_path=path)

    else:
        # use existing model

        model = Model.load_model(path)
        print(model.model.metrics_names)
        model.plot_results()

    return model


def prep_and_check_existing(path):
    # check if there is already an existing model at path
    use_existing = False

    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

    # if not os.path.exists(GRAPHS_DIR):
    #     os.mkdir(GRAPHS_DIR)

    if os.path.exists(path):
        use_existing = input("A trained model already exists: \nUse existing one? Y/N: ")
        use_existing = True if use_existing in YES else False

        if not use_existing:
            i = 1
            while os.path.exists('{}\\{}_Num{}'.format(MODELS_DIR, MODEL_FILE, i)):
                i += 1

            os.rename(path, '{}\\{}_Num{}'.format(MODELS_DIR, MODEL_FILE, i))

    return use_existing
