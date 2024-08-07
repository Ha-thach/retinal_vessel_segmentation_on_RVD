import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou

H = 512
W = 512

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    ori_x = x
    x = x / 255.0
    x = x.astype(np.float32)
    return ori_x, x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ori_x = x
    x = x / 255.0
    x = x.astype(np.int32)
    return ori_x, x

def load_data(path):
    x = sorted(glob(os.path.join(path, "Image", "*.png")))
    y = sorted(glob(os.path.join(path, "VeinMask", "*.png")))
    if not x or not y:
        print(f"No images or masks found in path: {path}")
    return x, y

def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    create_dir("/data/nhthach/project/RETINAL/training_vein_only(cropted)/result")

    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("/data/nhthach/project/RETINAL/TrainTestDataset(Crop&Augted)vein/model1.h5")

    dataset_path = os.path.join("/data/nhthach/project/RETINAL/DATA/TrainTestDataset(Crop&Augted)vein", "Test")
    test_x, test_y = load_data(dataset_path)

    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = x.split("/")[-1].split(".")[0]

        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)

        y_pred = model.predict(np.expand_dims(x, axis=0))[0]
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)
        y_pred = np.squeeze(y_pred, axis=-1)

        save_image_path = f"/data/nhthach/project/RETINAL/training_vein_only(cropted)/result/{name}.png"
        save_results(ori_x, ori_y, y_pred, save_image_path)

        y = y.flatten()
        y_pred = y_pred.flatten()

        if y.size == 0 or y_pred.size == 0:
            print(f"Skipping empty prediction for image {name}")
            continue

        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    if SCORE:
        score = np.mean([s[1:] for s in SCORE], axis=0)
        print(f"Accuracy: {score[0]:0.5f}")
        print(f"F1: {score[1]:0.5f}")
        print(f"Jaccard: {score[2]:0.5f}")
        print(f"Recall: {score[3]:0.5f}")
        print(f"Precision: {score[4]:0.5f}")
    else:
        print("No valid predictions were made.")

    df = pd.DataFrame(SCORE, columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("/data/nhthach/project/RETINAL/training_vein_only(cropted)/result/data1val.csv", index=False)
