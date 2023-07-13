import tensorflow as tf
import numpy as np
import pandas as pd
import gc
import psutil
import os
import random
from .constants import Constants

from Levenshtein import distance as Lev_distance


# Seed all random number generators
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def selected_columns(file_example):
    df = pd.read_parquet(file_example)
    selected_x = df.columns[[x + 1 for x in Constants.POINT_LANDMARKS]].tolist()
    selected_y = [c.replace("x", "y") for c in selected_x]
    selected = []
    for i in range(Constants.NUM_NODES):
        selected.append(selected_x[i])
        selected.append(selected_y[i])
    return selected  # x1,y1,x2,y2,...


class SWA(tf.keras.callbacks.Callback):
    # Stochastic Weight Averaging
    def __init__(
        self,
        save_name,
        swa_epochs=[],
        strategy=None,
        train_ds=None,
        valid_ds=None,
        train_steps=1000,
    ):
        super().__init__()
        self.swa_epochs = swa_epochs
        self.swa_weights = None
        self.save_name = save_name
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.train_steps = train_steps
        self.strategy = strategy

    # @tf.function(jit_compile=True)
    def train_step(self, iterator):
        """The step function for one training step."""

        def step_fn(inputs):
            """The computation to run on each device."""
            x, y = inputs
            _ = self.model(x, training=True)

        for x in iterator:
            self.strategy.run(step_fn, args=(x,))

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.swa_epochs:
            if self.swa_weights is None:
                self.swa_weights = self.model.get_weights()
            else:
                w = self.model.get_weights()
                for i in range(len(self.swa_weights)):
                    self.swa_weights[i] += w[i]

    def on_train_end(self, logs=None):
        if len(self.swa_epochs):
            print("applying SWA...")
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = self.swa_weights[i] / len(self.swa_epochs)
            self.model.set_weights(self.swa_weights)
            if self.train_ds is not None:  # for the re-calculation of running mean and var
                self.train_step(self.train_ds.take(self.train_steps))
            print(f"save SWA weights to {self.save_name}-SWA.h5")
            self.model.save_weights(f"{self.save_name}-SWA.h5")
            if self.valid_ds is not None:
                self.model.evaluate(self.valid_ds)


class AWP(tf.keras.Model):
    # Adversarial Weight Perturbation
    def __init__(self, *args, delta=0.1, eps=1e-4, start_step=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.eps = eps
        self.start_step = start_step

    def train_step_awp(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        params = self.trainable_variables
        params_gradients = tape.gradient(loss, self.trainable_variables)
        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]
            delta = tf.math.divide_no_nan(
                self.delta * grad, tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps
            )
            self.trainable_variables[i].assign_add(delta)
        with tf.GradientTape() as tape2:
            y_pred = self(x, training=True)
            new_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            if hasattr(self.optimizer, "get_scaled_loss"):
                new_loss = self.optimizer.get_scaled_loss(new_loss)

        gradients = tape2.gradient(new_loss, self.trainable_variables)
        if hasattr(self.optimizer, "get_unscaled_gradients"):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]
            delta = tf.math.divide_no_nan(
                self.delta * grad, tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps
            )
            self.trainable_variables[i].assign_sub(delta)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # self_loss.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return tf.cond(
            self._train_counter < self.start_step,
            lambda: super(AWP, self).train_step(data),
            lambda: self.train_step_awp(data),
        )


def num_to_char_fn(y):
    return [Constants.inv_dict.get(x, "") for x in y]


# A callback class to output a few transcriptions during training
class CallbackEval(tf.keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, model, dataset):
        super().__init__()
        self.dataset = dataset
        self.model = model

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = self.model(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = "".join(num_to_char_fn(label.numpy()))
                targets.append(label)
        print("-" * 100)
        # for i in np.random.randint(0, len(predictions), 2):
        for i in range(10):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}, len: {len(predictions[i])}")
            print("-" * 100)


def decode_phrase(pred):
    # decode cts prediction by prunning
    # (T,CHAR_NUMS)
    x = tf.argmax(pred, axis=1)
    paddings = tf.constant(
        [
            [0, 1],
        ]
    )
    x = tf.pad(x, paddings)
    diff = tf.not_equal(x[:-1], x[1:])
    adjacent_indices = tf.where(diff)[:, 0]
    x = tf.gather(x, adjacent_indices)
    mask = x != Constants.LABEL_PAD
    x = tf.boolean_mask(x, mask, axis=0)
    return x


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    output_text = []
    for result in pred:
        result = "".join(num_to_char_fn(decode_phrase(result).numpy()))
        output_text.append(result)
    return output_text


def calculate_N_D(s1, s2):
    length = len(s1)
    lvd = Lev_distance(s1, s2)
    return lvd, length


def code_to_label(label_code):
    label = [Constants.inv_dict[x] for x in label_code if Constants.inv_dict[x] != "P"]
    label = "".join(label)
    return label


def convert_to_strings(batch_label_code):
    output = []
    for label_code in batch_label_code:
        output.append(code_to_label(label_code))
    return output


def global_metric(val_ds, model):
    global_N, global_D = 0, 0
    count = 0
    metric = LevDistanceMetric()
    for batch in val_ds:
        count += 1
        print(count)
        feature, label = batch
        logits = model(feature)
        _, _, D = batch_edit_distance(label, logits)
        metric.update_state(label, logits)

        label = label.numpy()
        target_strings = convert_to_strings(label)
        predict_strings = decode_batch_predictions(logits)

        values = [
            calculate_N_D(target, predict)
            for target, predict in zip(target_strings, predict_strings)
        ]
        batch_D = np.sum([x[0] for x in values])
        batch_N = np.sum([x[1] for x in values])
        global_D += batch_D
        global_N += batch_N
    metric_value = np.clip((global_N - global_D) / global_N, a_min=0, a_max=1)
    result = metric.result().numpy()
    print("Custom metric", result)
    print("External Lev package", metric_value)
    return metric_value


def sparse_from_dense_ignore_value(dense_tensor):
    mask = tf.not_equal(dense_tensor, Constants.LABEL_PAD)
    indices = tf.where(mask)
    values = tf.boolean_mask(dense_tensor, mask)
    return tf.SparseTensor(indices, values, tf.shape(dense_tensor, out_type=tf.int64))


def batch_edit_distance(y_true, y_logits):
    blank = Constants.LABEL_PAD
    B = tf.shape(y_logits)[0]
    seq_length = tf.shape(y_logits)[1]
    to_decode = tf.transpose(y_logits, perm=[1, 0, 2])
    sequence_length = tf.fill(dims=[B], value=seq_length)
    hypothesis = tf.nn.ctc_greedy_decoder(
        tf.cast(to_decode, tf.float32), sequence_length, blank_index=blank
    )[0][
        0
    ]  # full is [B,...]
    truth = sparse_from_dense_ignore_value(y_true)  # full is [B,...]

    edit_dist = tf.edit_distance(hypothesis, truth, normalize=False)

    non_ignore_mask = tf.not_equal(y_true, blank)
    N = tf.reduce_sum(tf.cast(non_ignore_mask, tf.float32))
    D = tf.reduce_sum(edit_dist)
    result = (N - D) / N
    result = tf.clip_by_value(result, 0.0, 1.0)
    return result, N, D


class LevDistanceMetric(tf.keras.metrics.Metric):
    def __init__(self, name="Lev", **kwargs):
        super().__init__(name=name, **kwargs)
        self.distance = self.add_weight(name="dist", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_logits, sample_weight=None):
        # if using with keras compile, make sure the model outputs logits, not softmax probabilities
        _, N, D = batch_edit_distance(y_true, y_logits)
        self.distance.assign_add(D)
        self.count.assign_add(N)

    def result(self):
        result = (self.count - self.distance) / self.count
        result = tf.clip_by_value(result, 0.0, 1.0)
        return result

    def reset_state(self):
        self.count.assign(0.0)
        self.distance.assign(0.0)


class MemoryUsageCallbackExtended(tf.keras.callbacks.Callback):
    """Monitor memory usage on epoch begin and end, collect garbage"""

    # def on_epoch_begin(self, epoch, logs=None):
    #    print("**Epoch {}**".format(epoch))
    #    print(
    #        f"Memory usage on epoch begin: {int(psutil.Process(os.getpid()).memory_info().rss)/1e9:.2f GB}"
    #    )

    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Memory usage on epoch end: {int(psutil.Process(os.getpid()).memory_info().rss)/1e9:.2f} GB"
        )
        gc.collect()
