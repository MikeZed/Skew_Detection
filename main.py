

from model_creation import create_model
from image_utils import correct_image_using_model

YES = ['y', 'Y', '1', ' ']
SAVE_MODEL = True

## DATA ##
URL = "http://www.iit.demokritos.gr/~alexpap/DISEC13/icdar2013_benchmarking_dataset.rar"
DATA_PATH = (URL.split("/")[-1]).replace(".rar", "")
DATA_FILE = "Ground_Truth.txt"

USE_GENERATOR = False
TRAIN_VAL_TEST_SPLIT = (60, 20, 20)

## Transfer Learning  ##
# USE_TRANSFER = False
# MODEL_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

settings = {'save_model': SAVE_MODEL, 'use_generator': USE_GENERATOR}


def main():
    SkewEstimator = create_model(URL, DATA_PATH, DATA_FILE, split=TRAIN_VAL_TEST_SPLIT, **settings)

    while True:

        img_num = input("Image Num to correct (Any number from 1 to 1550): ")

        img_num = img_num.split(' ')
        img_num = list(map(float, img_num))

        try:
            corrected_image = \
                correct_image_using_model(SkewEstimator,
                                          "SampleSet\IMG({:03d})_SA[{:.2f}].tif".format(int(img_num[0]), img_num[1]))
        # correct_image_using_model(model, "{}\\IMG_{:04d}.tif".format(DATA_PATH, int(img_num)))

        except:
            break

    print("Finishing program...")


if __name__ == "__main__":
    main()
