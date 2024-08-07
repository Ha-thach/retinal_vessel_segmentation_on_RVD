import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import build_unet
from metrics import dice_loss, dice_coef, iou

print("Cuda confirmation:", tf.test.is_built_with_cuda())
print("Available GPU devices:", tf.config.list_physical_devices('GPU'))

H = 512
W = 512

# Save weights every 10 epochs
# class SaveWeightsCallback(tf.keras.callbacks.Callback):
#     def __init__(self, model_path_template):
#         super(SaveWeightsCallback, self).__init__()
#         self.model_path_template = model_path_template
#
#     def on_epoch_end(self, epoch, logs=None):
#         if (epoch + 1) % 10 == 0:
#             self.model.save_weights(self.model_path_template.format(epoch=epoch + 1))
#             print(f"Saved weights at epoch {epoch + 1}")

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    x = sorted(glob(os.path.join(path, "Image", "*.png")))
    y = sorted(glob(os.path.join(path, "VeinMask", "*.png")))
    return x, y

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    #x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(path):
    path = path.decode()
    y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    #y = cv2.resize(y, (W, H))
    y = y / 255.0
    y = y.astype(np.float32)
    y = np.expand_dims(y, axis=-1)  ## (512, 512, 1)
    return y


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def generator(X, Y):
    for x, y in zip(X, Y):
        yield x, y

def tf_dataset(X, Y, batch_size=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    # input_signature = (tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32), tf.TensorSpec(shape=(512, 512, 1), dtype=tf.float32))
    # dataset = tf.data.Dataset.from_generator(lambda: generator(X, Y), output_signature=input_signature)
    # dataset = tf.data.Dataset.from_generator(lambda: generator(X, Y), output_types = ('float32', 'float32'), output_shapes = (tf.TensorShape((1, )), tf.TensorShape((1,))))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory to save files """
    create_dir("/data/nhthach/project/RETINAL/training_vein_only(cropted)")

    """ Hyperparameters """
    batch_size = 2
    lr = 1e-4
    num_epochs = 20
    model_path = os.path.join("/data/nhthach/project/RETINAL/TrainTestDataset(Crop&Augted)vein", "model3.h5")
    csv_path = os.path.join("/data/nhthach/project/RETINAL/TrainTestDataset(Crop&Augted)vein", "data3.csv")

    """ Dataset """
    dataset_path = "/data/nhthach/project/RETINAL/DATA/TrainTestDataset(Crop&Augted)vein"
    train_path = os.path.join(dataset_path, "Train")
    valid_path = os.path.join(dataset_path, "Test")

    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch_size=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=batch_size)

    train_steps = len(train_x) // batch_size
    valid_steps = len(valid_x) // batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1
    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    """ Model """
    model = build_unet((H, W, 3))
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision()])

    # model.summary()

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False)]
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )
