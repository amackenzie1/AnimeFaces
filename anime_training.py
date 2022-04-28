import tensorflow as tf 
import numpy as np
import random
import datetime
from anime_model import get_model
from anime_dataset_gen import generator
import os 

def fit_fn():
        dataset = tf.data.Dataset.from_generator(generator, output_types=({"input" : tf.float32, "time": tf.float32}, tf.float32))
        dataset = dataset.batch(32)
        return dataset

working_dir = "tmp"
log_dir = os.path.join(working_dir, "log")
ckpt_filepath = os.path.join(working_dir, "ckpt")
backup_dir = os.path.join(working_dir, "backup")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath),
    tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir),
]

if __name__ == "__main__":
        model = get_model()
        lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 2000, 0.95)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        # model.load_weights("weights/variables")
        # model = tf.keras.models.load_model("alt/v3")
        with tf.device("gpu:0"):
            model.fit(fit_fn(), epochs=1000, steps_per_epoch=8192//32, callbacks=callbacks)
            model.save_weights("weights/v3")
            model.save("alt/v3")

