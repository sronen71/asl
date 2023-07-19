#!/usr/bin/env python
# coding: utf-8

# # Google - American Sign Language Fingerspelling Recognition with TensorFlow
# 
# This notebook walks you through how to train a Transformer model using TensorFlow on the Google - American Sign Language Fingerspelling Recognition dataset made available for this competition.
# 
# The objective of the model is to predict and translate American Sign Language (ASL) fingerspelling from a set of video frames into text(`phrase`).
# 
# In this notebook you will learn:
# 
# - How to load the data
# - Convert the data to tfrecords to make it faster to re-traing the model
# - Train a transformer models on the data
# - Convert the model to TFLite
# - Create a submission

# # Installation
# 
# Specifically for this competition, you'll need the mediapipe library to work on the data and visualize it

# # Import the libraries

# In[ ]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import psutil
import glob
import gc
import math
from tensorflow.python.framework.ops import disable_eager_execution


# In[ ]:


gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[ ]:


print("TensorFlow v" + tf.__version__)


# In[ ]:


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


# # Scheduler

# In[ ]:


class CosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay with optional warmup.

    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.

    For the idea of a linear warmup of our learning rate,
    see [Goyal et al.](https://arxiv.org/pdf/1706.02677.pdf).

    When we begin training a model, we often want an initial increase in our
    learning rate followed by a decay. If `warmup_target` is an int, this
    schedule applies a linear increase per optimizer step to our learning rate
    from `initial_learning_rate` to `warmup_target` for a duration of
    `warmup_steps`. Afterwards, it applies a cosine decay function taking our
    learning rate from `warmup_target` to `alpha` for a duration of
    `decay_steps`. If `warmup_target` is None we skip warmup and our decay
    will take our learning rate from `initial_learning_rate` to `alpha`.
    It requires a `step` value to  compute the learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.

    The schedule is a 1-arg callable that produces a warmup followed by a
    decayed learning rate when passed the current optimizer step. This can be
    useful for changing the learning rate value across different invocations of
    optimizer functions.

    Our warmup is computed as:

    ```python
    def warmup_learning_rate(step):
        completed_fraction = step / warmup_steps
        total_delta = target_warmup - initial_learning_rate
        return completed_fraction * total_delta
    ```

    And our decay is computed as:

    ```python
    if warmup_target is None:
        initial_decay_lr = initial_learning_rate
    else:
        initial_decay_lr = warmup_target

    def decayed_learning_rate(step):
        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return initial_decay_lr * decayed
    ```

    Example usage without warmup:

    ```python
    decay_steps = 1000
    initial_learning_rate = 0.1
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps)
    ```

    Example usage with warmup:

    ```python
    decay_steps = 1000
    initial_learning_rate = 0
    warmup_steps = 1000
    target_learning_rate = 0.1
    lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
        warmup_steps=warmup_steps
    )
    ```

    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.

    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        alpha=0.0,
        name=None,
        warmup_target=None,
        warmup_steps=0,
    ):
        """Applies cosine decay to the learning rate.

        Args:
          initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python int. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python int.
            Number of steps to decay over.
          alpha: A scalar `float32` or `float64` `Tensor` or a Python int.
            Minimum learning rate value for decay as a fraction of
            `initial_learning_rate`.
          name: String. Optional name of the operation.  Defaults to
            'CosineDecay'.
          warmup_target: None or a scalar `float32` or `float64` `Tensor` or a
            Python int. The target learning rate for our warmup phase. Will cast
            to the `initial_learning_rate` datatype. Setting to None will skip
            warmup and begins decay phase from `initial_learning_rate`.
            Otherwise scheduler will warmup from `initial_learning_rate` to
            `warmup_target`.
          warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python int.
            Number of steps to warmup over.
        """
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name
        self.warmup_steps = warmup_steps
        self.warmup_target = warmup_target

    def _decay_function(self, step, decay_steps, decay_from_lr, dtype):
        with tf.name_scope(self.name or "CosineDecay"):
            completed_fraction = step / decay_steps
            tf_pi = tf.constant(math.pi, dtype=dtype)
            cosine_decayed = 0.5 * (1.0 + tf.cos(tf_pi * completed_fraction))
            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            return tf.multiply(decay_from_lr, decayed)

    def _warmup_function(self, step, warmup_steps, warmup_target, initial_learning_rate):
        with tf.name_scope(self.name or "CosineDecay"):
            completed_fraction = step / warmup_steps
            total_step_delta = warmup_target - initial_learning_rate
            return total_step_delta * completed_fraction + initial_learning_rate

    def __call__(self, step):
        with tf.name_scope(self.name or "CosineDecay"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            global_step_recomp = tf.cast(step, dtype)

            if self.warmup_target is None:
                global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
                return self._decay_function(
                    global_step_recomp,
                    decay_steps,
                    initial_learning_rate,
                    dtype,
                )

            warmup_target = tf.cast(self.warmup_target, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)

            global_step_recomp = tf.minimum(global_step_recomp, decay_steps + warmup_steps)

            return tf.cond(
                global_step_recomp < warmup_steps,
                lambda: self._warmup_function(
                    global_step_recomp,
                    warmup_steps,
                    warmup_target,
                    initial_learning_rate,
                ),
                lambda: self._decay_function(
                    global_step_recomp - warmup_steps,
                    decay_steps,
                    warmup_target,
                    dtype,
                ),
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name,
            "warmup_target": self.warmup_target,
            "warmup_steps": self.warmup_steps,
        }


# # Constants

# In[ ]:


def get_char_dict():
    char_dict = {
        " ": 0,
        "!": 1,
        "#": 2,
        "$": 3,
        "%": 4,
        "&": 5,
        "'": 6,
        "(": 7,
        ")": 8,
        "*": 9,
        "+": 10,
        ",": 11,
        "-": 12,
        ".": 13,
        "/": 14,
        "0": 15,
        "1": 16,
        "2": 17,
        "3": 18,
        "4": 19,
        "5": 20,
        "6": 21,
        "7": 22,
        "8": 23,
        "9": 24,
        ":": 25,
        ";": 26,
        "=": 27,
        "?": 28,
        "@": 29,
        "[": 30,
        "_": 31,
        "a": 32,
        "b": 33,
        "c": 34,
        "d": 35,
        "e": 36,
        "f": 37,
        "g": 38,
        "h": 39,
        "i": 40,
        "j": 41,
        "k": 42,
        "l": 43,
        "m": 44,
        "n": 45,
        "o": 46,
        "p": 47,
        "q": 48,
        "r": 49,
        "s": 50,
        "t": 51,
        "u": 52,
        "v": 53,
        "w": 54,
        "x": 55,
        "y": 56,
        "z": 57,
        "~": 58,
    }
    char_dict["P"] = 59
    char_dict["SOS"] = 60
    char_dict["EOS"] = 61
    return char_dict


class Constants:
    ROWS_PER_FRAME = 543
    MAX_STRING_LEN = 50
    INPUT_PAD = -100.0
    char_dict = get_char_dict()
    LABEL_PAD = char_dict["P"]
    inv_dict = {v: k for k, v in char_dict.items()}
    NOSE = [1, 2, 98, 327]

    REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
    LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]

    LHAND = list(range(468, 489))
    RHAND = list(range(522, 543))

    LNOSE = [98]
    RNOSE = [327]

    LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
    RLIP = [
        314,
        405,
        321,
        375,
        291,
        409,
        270,
        269,
        267,
        317,
        402,
        318,
        324,
        308,
        415,
        310,
        311,
        312,
    ]
    POSE = [500, 502, 504, 501, 503, 505, 512, 513]
    LPOSE = [513, 505, 503, 501]
    RPOSE = [512, 504, 502, 500]

    POINT_LANDMARKS_PARTS = [LHAND, RHAND, LLIP, RLIP, LPOSE, RPOSE, NOSE, REYE, LEYE]
    # POINT_LANDMARKS_PARTS = [LHAND, RHAND, NOSE]
    POINT_LANDMARKS = [item for sublist in POINT_LANDMARKS_PARTS for item in sublist]
    parts = {
        "LLIP": LLIP,
        "RLIP": RLIP,
        "LHAND": LHAND,
        "RHAND": RHAND,
        "LPOSE": LPOSE,
        "RPOSE": RPOSE,
        "LNOSE": LNOSE,
        "RNOSE": RNOSE,
        "REYE": REYE,
        "LEYE": LEYE,
    }

    LANDMARK_INDICES = {}  # type: ignore
    for part in parts:
        LANDMARK_INDICES[part] = []
        for landmark in parts[part]:
            if landmark in POINT_LANDMARKS:
                LANDMARK_INDICES[part].append(POINT_LANDMARKS.index(landmark))

    CENTER_LANDMARKS = LNOSE + RNOSE
    CENTER_INDICES = LANDMARK_INDICES["LNOSE"] + LANDMARK_INDICES["RNOSE"]

    NUM_NODES = len(POINT_LANDMARKS)
    NUM_INPUT_FEATURES = 2 * NUM_NODES
    CHANNELS = 6 * NUM_NODES


# # Utils

# In[ ]:


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
   
    result = metric.result().numpy()
   
    return result


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
    truth = tf.cast(truth, tf.int64)
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


# # Model

# In[ ]:


class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, pad_token_idx):
        self.pad_token_idx = pad_token_idx
        super().__init__()

    def call(self, labels, logits):
        label_length = tf.reduce_sum(tf.cast(labels != self.pad_token_idx, tf.int32), axis=-1)
        logit_length = tf.ones(tf.shape(logits)[0], dtype=tf.int32) * tf.shape(logits)[1]

        ctc_loss = tf.nn.ctc_loss(
            labels=labels,
            logits=logits,
            label_length=label_length,
            logit_length=logit_length,
            blank_index=self.pad_token_idx,
            logits_time_major=False,
        )

        return ctc_loss


class ECA(tf.keras.layers.Layer):
    # Efficient Channel Attention
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(
            1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False
        )

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn


class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)

    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        x = tf.cond(
            self._train_counter < self.start_step,
            lambda: inputs,
            lambda: self.dropout(inputs, training=training),
        )
        if training:
            self._train_counter.assign_add(1)
        return x


class CausalDWConv1D(tf.keras.layers.Layer):
    # Causal Depth Wise Convolution
    def __init__(
        self,
        kernel_size=17,
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer="glorot_uniform",
        name="",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D(
            (dilation_rate * (kernel_size - 1), 0), name=name + "_pad"
        )
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size,
            strides=1,
            dilation_rate=dilation_rate,
            padding="valid",
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            name=name + "_dwconv",
        )
        self.supports_masking = True

    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x


def Conv1DBlock(
    channel_size,
    kernel_size,
    dilation_rate=1,
    drop_rate=0.0,
    expand_ratio=2,
    # se_ratio=0.25,
    activation="swish",
    name=None,
):
    """
    efficient conv1d block, @hoyso48
    """
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))

    # Expansion phase
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(
            channels_expand, use_bias=True, activation=activation, name=name + "_expand_conv"
        )(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(
            kernel_size, dilation_rate=dilation_rate, use_bias=False, name=name + "_dwconv"
        )(x)

        x = tf.keras.layers.LayerNormalization(name=name + "_bn")(x)

        x = ECA()(x)  # efficient channel attention

        x = tf.keras.layers.Dense(channel_size, use_bias=True, name=name + "_project_conv")(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "_drop")(x)

        if channels_in == channel_size:
            x = tf.keras.layers.add([x, skip], name=name + "_add")
        return x

    return apply


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim**-0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(
            tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv)
        )
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x


def TransformerBlock(
    dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation="swish"
):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.LayerNormalization()(x)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x

    return apply


def build_model1(
    output_dim,
    max_len=64,
    dropout_step=0,
    dim=192,
    input_pad=-100,
    with_transformer=False,
    drop_rate=0.2,
):
    inp = tf.keras.Input(shape=(max_len, Constants.CHANNELS), dtype=tf.float32, name="inputs")
    x = tf.keras.layers.Masking(mask_value=input_pad, input_shape=(max_len, Constants.CHANNELS))(
        inp
    )
    ksize = 17
    x = tf.keras.layers.Dense(dim, use_bias=False, name="stem_conv")(x)
    x = tf.keras.layers.LayerNormalization(name="stem_bn")(x)

    x = Conv1DBlock(dim, ksize, drop_rate=drop_rate)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=drop_rate)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=drop_rate)(x)
    if with_transformer:
        x = TransformerBlock(dim, expand=2)(x)

    x = tf.keras.layers.AvgPool1D(2, 2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=drop_rate)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=drop_rate)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=drop_rate)(x)
    if with_transformer:
        x = TransformerBlock(dim, expand=2)(x)

    if dim == 384:  # for the 4x sized model
        x = Conv1DBlock(dim, ksize, drop_rate=drop_rate)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=drop_rate)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=drop_rate)(x)
        if with_transformer:
            x = TransformerBlock(dim, expand=2)(x)

        x = Conv1DBlock(dim, ksize, drop_rate=drop_rate)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=drop_rate)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=drop_rate)(x)
        if with_transformer:
            x = TransformerBlock(dim, expand=2)(x)

    lstm = tf.keras.layers.LSTM(units=output_dim, return_sequences=True)
    x = tf.keras.layers.Bidirectional(lstm)(x)
    # x = LateDropout(0.8, start_step=dropout_step)(x)
    # x = tf.keras.layers.LayerNormalization()(x)

    outputs = tf.keras.layers.Dense(output_dim, activation="log_softmax")(x)  # logits

    # x = tf.keras.layers.Dense(output_dim)(x)  # logits
    # outputs = tf.keras.layers.Activation("log_softmax", dtype="float32")(x)
    model = tf.keras.Model(inp, outputs)
    return model


def get_model(output_dim, max_len, dim, input_pad):
   
    model = build_model1(output_dim, max_len=max_len, input_pad=input_pad, dim=dim)
    return model


# # Configuration

# In[ ]:


def get_strategy():
    logical_devices = tf.config.list_logical_devices()
    # Check if TPU is available

    gpu_available = any("GPU" in device.name for device in logical_devices)
    strategy = None
    is_tpu = False
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU ", tpu.master())
        is_tpu = True
    except ValueError:
        is_tpu = False

    if is_tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        disable_eager_execution()  # LSTM layer can't use bfloat16 unless we do this.

    else:
        if gpu_available:
          
            ngpu = len(gpus)
            print("Num GPUs Available: ", ngpu)
            if ngpu > 1:
                strategy = tf.distribute.MirroredStrategy()
            else:
                strategy = tf.distribute.get_strategy()

        else:
            print("Runing on CPU")
            strategy = tf.distribute.get_strategy()
    replicas = strategy.num_replicas_in_sync

    print(f"get strategy replicas: {replicas}")

    return strategy, replicas, is_tpu


class CFG:
    # These 3 variables are update dynamically later by calling update_config_with_strategy.
    strategy = None  # type: ignore
    replicas = 1
    is_tpu = False

    save_output = True
    log_path = "/kaggle/working/"
    input_path = "/kaggle/input/asl-fingerspelling/"
    output_path = "/kaggle/working/"

    seed = 42
    verbose = 1  # 0) silent 1) progress bar 2) one line per epoch

    # max number of frames
    #max_len = 256
    max_len = 256
    replicas = 1
    
    lr = 5e-4   # 5e-4
    weight_decay = 1e-4  # 4e-4
    epochs = 80 
    batch_size=128
    snapshot_epochs = []  # type: ignore
    swa_epochs = []  # type: ignore
    # list(range(epoch//2,epoch+1))

    fp16 = True
    
    awp = False
    awp_lambda = 0.2
    awp_start_epoch = 15
    dropout_start_epoch = 15
    resume = 0
    
    dim = 384
    
    comment = f"model-{dim}-seed{seed}"
    output_dim = 61
    num_eval = 6




# In[ ]:


def update_config_with_strategy(config):
    # cfg is configuration instance
    strategy, replicas, is_tpu = get_strategy()
    config.strategy = strategy
    config.replicas = replicas
    config.is_tpu = is_tpu
    config.lr = config.lr * replicas
    config.batch_size = config.batch_size * replicas
    return config


# # Training

# In[ ]:





# In[ ]:


def count_data_items(dataset):
    dataset_size = 0
    for _ in dataset:
        dataset_size += 1
    return dataset_size


def interp1d_(x, target_len):
    target_len = tf.maximum(1, target_len)
    x = tf.image.resize(x, (target_len, tf.shape(x)[1]))
    return x


def tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims
    ) / tf.reduce_sum(
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims
    )


def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))


def flip_lr(x):
    if x.shape[1] == Constants.ROWS_PER_FRAME:
        LHAND = Constants.LHAND
        RHAND = Constants.RHAND
        LLIP = Constants.LLIP
        RLIP = Constants.RLIP
        LEYE = Constants.LEYE
        REYE = Constants.REYE
        LNOSE = Constants.LNOSE
        RNOSE = Constants.RNOSE
        LPOSE = Constants.LPOSE
        RPOSE = Constants.RPOSE
    else:
        LHAND = Constants.LANDMARK_INDICES["LHAND"]
        RHAND = Constants.LANDMARK_INDICES["RHAND"]
        LLIP = Constants.LANDMARK_INDICES["LLIP"]
        RLIP = Constants.LANDMARK_INDICES["RLIP"]
        LEYE = Constants.LANDMARK_INDICES["LEYE"]
        REYE = Constants.LANDMARK_INDICES["REYE"]
        LNOSE = Constants.LANDMARK_INDICES["LNOSE"]
        RNOSE = Constants.LANDMARK_INDICES["RNOSE"]
        LPOSE = Constants.LANDMARK_INDICES["LPOSE"]
        RPOSE = Constants.LANDMARK_INDICES["RPOSE"]

    x, y = tf.unstack(x, axis=-1)
    x = 1 - x
    new_x = tf.stack([x, y], -1)
    new_x = tf.transpose(new_x, [1, 0, 2])
    lhand = tf.gather(new_x, LHAND, axis=0)
    rhand = tf.gather(new_x, RHAND, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LHAND)[..., None], rhand)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RHAND)[..., None], lhand)
    llip = tf.gather(new_x, LLIP, axis=0)
    rlip = tf.gather(new_x, RLIP, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LLIP)[..., None], rlip)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RLIP)[..., None], llip)
    lpose = tf.gather(new_x, LPOSE, axis=0)
    rpose = tf.gather(new_x, RPOSE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LPOSE)[..., None], rpose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RPOSE)[..., None], lpose)
    leye = tf.gather(new_x, LEYE, axis=0)
    reye = tf.gather(new_x, REYE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LEYE)[..., None], reye)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(REYE)[..., None], leye)
    lnose = tf.gather(new_x, LNOSE, axis=0)
    rnose = tf.gather(new_x, RNOSE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LNOSE)[..., None], rnose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RNOSE)[..., None], lnose)
    new_x = tf.transpose(new_x, [1, 0, 2])
    return new_x


def resample(x, rate=(0.8, 1.2)):
    rate = tf.random.uniform((), rate[0], rate[1])
    length = tf.shape(x)[0]
    new_size = tf.cast(rate * tf.cast(length, tf.float32), tf.int32)
    new_x = interp1d_(x, new_size)
    return new_x


def spatial_random_affine(
    xyz,
    scale=(0.8, 1.2),
    shear=(-0.1, 0.1),
    shift=(-0.1, 0.1),
    degree=(-20, 20),
):
    center = tf.constant([0.5, 0.5])
    if degree is not None:
        xy = xyz[..., :2]
        z = xyz[..., 2:]
        xy -= center
        degree = tf.random.uniform((), *degree)
        radian = degree / 180 * np.pi
        c = tf.math.cos(radian)
        s = tf.math.sin(radian)
        rotate_mat = tf.identity(
            [
                [c, s],
                [-s, c],
            ]
        )
        xy = xy @ rotate_mat
        xy = xy + center
        xyz = tf.concat([xy, z], axis=-1)

    if scale is not None:
        scale = tf.random.uniform((), *scale)
        xyz = scale * xyz

    if shear is not None:
        xy = xyz[..., :2]
        z = xyz[..., 2:]
        shear_x = shear_y = tf.random.uniform((), *shear)
        if tf.random.uniform(()) < 0.5:
            shear_x = 0.0
        else:
            shear_y = 0.0
        shear_mat = tf.identity([[1.0, shear_x], [shear_y, 1.0]])
        xy = xy @ shear_mat
        xyz = tf.concat([xy, z], axis=-1)

    if shift is not None:
        shift = tf.random.uniform((), *shift)
        xyz = xyz + shift

    return xyz


def temporal_mask(x, size=[1, 10], mask_value=float("nan")):
    l0 = tf.shape(x)[0]
    if size[1] > l0 // 8:
        size[1] = l0 // 8
        if size[1] <= 1:
            size[1] = 2
    mask_size = tf.random.uniform((), *size, dtype=tf.int32)
    mask_offset = tf.random.uniform((), 0, tf.clip_by_value(l0 - mask_size, 1, l0), dtype=tf.int32)
    x = tf.tensor_scatter_nd_update(
        x,
        tf.range(mask_offset, mask_offset + mask_size)[..., None],
        tf.fill([mask_size, tf.shape(x)[1], 2], mask_value),
    )
    return x


def spatial_mask(x, size=(0.05, 0.2), mask_value=float("nan")):
    mask_offset_y = tf.random.uniform(())
    mask_offset_x = tf.random.uniform(())
    mask_size = tf.random.uniform((), *size)
    mask_x = (mask_offset_x < x[..., 0]) & (x[..., 0] < mask_offset_x + mask_size)
    mask_y = (mask_offset_y < x[..., 1]) & (x[..., 1] < mask_offset_y + mask_size)
    mask = mask_x & mask_y
    x = tf.where(mask[..., None], mask_value, x)
    return x


#@tf.function()
def augment_fn(x):
    # shape (T,F)
    x = tf.reshape(x, (tf.shape(x)[0], -1, 2))
    if tf.random.uniform(()) < 0.4:
        x = resample(x, (0.5, 1.5))
    if tf.random.uniform(()) < 0.4:
        x = flip_lr(x)
    if tf.random.uniform(()) < 0.4:
        x = spatial_random_affine(x)
    if tf.random.uniform(()) < 0.2:
        x = temporal_mask(x)
    if tf.random.uniform(()) < 0.2:
        x = spatial_mask(x)
    x = tf.reshape(x, (tf.shape(x)[0], -1))
    return x


# In[ ]:


class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len, normalize=False, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.center = Constants.CENTER_INDICES
        self.normalize = normalize

    # preprocess a batch of data
    def call(self, x):
        # rank is 3: [B,T,F]
        # if your input is just [T,F], extend its dimesnion before calling.

        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1, 2))
        # dimensions now are [B,T,F//2,2]

        x_selected = x
        if self.normalize:
            mean = tf_nan_mean(tf.gather(x, self.center, axis=2), axis=[1, 2], keepdims=True)
            mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
            std = tf_nan_std(x_selected, center=mean, axis=[1, 2], keepdims=True)
            x = (x_selected - mean) / std
        else:
            x = x_selected

        dx = tf.cond(
            tf.shape(x)[1] > 1,
            lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]),
            lambda: tf.zeros_like(x),
        )

        dx2 = tf.cond(
            tf.shape(x)[1] > 2,
            lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]),
            lambda: tf.zeros_like(x),
        )
        length = tf.shape(x)[1]

        x = tf.concat(
            [
                tf.reshape(x, (-1, length, 2 * Constants.NUM_NODES)),  # x1,y1,x2,y2,...
                tf.reshape(dx, (-1, length, 2 * Constants.NUM_NODES)),
                tf.reshape(dx2, (-1, length, 2 * Constants.NUM_NODES)),
            ],
            axis=-1,
        )

        # x1,y1,x2,y2,...dx1,dy1,dx2,dy2,...
        x = tf.where(tf.math.is_nan(x), tf.constant(0.0, x.dtype), x)
        return x


def pad_if_short(x, max_len):
    # shape (T,F)
    pad_len = max_len - tf.shape(x)[0]
    padding = tf.ones((pad_len, tf.shape(x)[1]), dtype=x.dtype) * Constants.INPUT_PAD
    x = tf.concat([x, padding], axis=0)
    return x


def shrink_if_long(x, max_len):
    # shape is [T,F]
    if tf.shape(x)[0] > max_len:
        # we need to extend the dimension to [T,F,channels]  for tf.image.resize
        x = tf.image.resize(x[..., None], (max_len, tf.shape(x)[1]))
        x = tf.squeeze(x, axis=2)
    return x

def preprocess(x, max_len, do_pad=True):
    # shape (T,F)
    x = shrink_if_long(x, max_len=max_len)
    # Preprocess expects a batch, so we extend the dimension to (None,T,F), then reduce the output back to (T,F).
    x = tf.cast(Preprocess(max_len=max_len)(x[None, ...])[0], tf.float32)

    if do_pad:  # we can avoid this step if there is batch padding
        x = pad_if_short(x, max_len=max_len)

    return x


# In[ ]:


def decode_tfrec(record_bytes):
    features = tf.io.parse_single_example(
        record_bytes,
        {
            "coordinates": tf.io.VarLenFeature(tf.float32),
            "label": tf.io.VarLenFeature(tf.int64),
        },
    )
    coords = tf.sparse.to_dense(features["coordinates"])
    coords = tf.reshape(coords, (-1, Constants.NUM_INPUT_FEATURES))
    label = tf.sparse.to_dense(features["label"])

    return (coords, label)


# In[ ]:


def get_dataset(
    filenames,
    input_path,
    max_len,
    batch_size=64,
    drop_remainder=False,
    augment=False,
    shuffle_buffer=None,
    repeat=False,
    use_tfrecords=True,
):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    
    ds = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=tf.data.AUTOTUNE, compression_type="GZIP"
    )
    ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
  
    ds.with_options(ignore_order)
    
    if augment:
        ds = ds.map(lambda x, y: (augment_fn(x), y), tf.data.AUTOTUNE)
    
    ds = ds.map(lambda x, y: (preprocess(x, max_len=max_len, do_pad=False), y), tf.data.AUTOTUNE)
    #if repeat:
    #    ds = ds.repeat()
    
    if shuffle_buffer is not None:
        ds = ds.shuffle(shuffle_buffer)
  
    ds = ds.padded_batch(
        batch_size,
        padding_values=(
            tf.constant(Constants.INPUT_PAD, dtype=tf.float32),
            tf.constant(Constants.LABEL_PAD, dtype=tf.int64),
        ),
        padded_shapes=([max_len, Constants.CHANNELS], [Constants.MAX_STRING_LEN]),
        drop_remainder=drop_remainder,
    )
    
    #tf.data.experimental.assert_cardinality(len(labels) // BATCH_SIZE)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# In[ ]:


def train_run(train_files, valid_files, config, num_train, experiment_id=0, use_tfrecords=True,summary=False):
    gc.collect()
    tf.keras.backend.clear_session()
    # tf.config.optimizer.set_jit("autoclustering")

    if config.fp16:
        if config.is_tpu:
            policy = "mixed_bfloat16"
        else:
            policy = "mixed_float16"
    else:
        policy = "float32"
    tf.keras.mixed_precision.set_global_policy(policy)

    augment_train= True
    repeat_train = True

    shuffle_buffer = 4096
    train_ds = get_dataset(
        train_files,
        input_path=config.input_path,
        max_len=config.max_len,
        batch_size=config.batch_size,
        drop_remainder=True,
        augment=augment_train,
        repeat=repeat_train,
        shuffle_buffer=shuffle_buffer,
        use_tfrecords=True,
    )
    if valid_files is not None:
        valid_ds = get_dataset(
            valid_files,
            input_path=config.input_path,
            max_len=config.max_len,
            batch_size=config.batch_size,
            use_tfrecords=True,
        )
    else:
        valid_ds = None
        valid_files = []

    # num_train = count_data_items(train_ds)
    # num_valid = count_data_items(valid_ds)
    # print(num_train, num_valid, config.batch_size)
    # exit()

    steps_per_epoch = num_train // config.batch_size
    strategy = config.strategy
    with strategy.scope():
        model = get_model(
            max_len=config.max_len,
            output_dim=config.output_dim,
            input_pad=Constants.INPUT_PAD,
            dim=config.dim,
        )
 
        base_lr = config.lr
        lr_schedule = CosineDecay(
            initial_learning_rate=base_lr / 10,
            decay_steps=int(0.95 * steps_per_epoch * config.epochs),
            alpha=0.02,
            name=None,
            warmup_target=base_lr,
            warmup_steps=int(0.05 * steps_per_epoch * config.epochs),
        )
        
        opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=config.weight_decay)
        #opt = tf.keras.optimizers.AdamW(learning_rate=2e-4)
        loss = CTCLoss(pad_token_idx=Constants.LABEL_PAD)

        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=[
                LevDistanceMetric(),
            ],
            #jit_compile= not config.is_tpu # Should be False on TPU!!
        )

    if summary:
        print()
        model.summary()
        print()
        print(train_ds, valid_ds)
        print()
    print(f"---------experiment {experiment_id}---------")
    print(f"train:{num_train} ")
    print()

    if config.resume:
        print(f"resume from epoch{config.resume}")
        model.load_weights(f"{config.log_path}/{config.comment}-exp{experiment_id}-last.h5")
        if train_ds is not None:
            model.evaluate(train_ds.take(steps_per_epoch))
        if valid_ds is not None:
            model.evaluate(valid_ds)

    tb_logger = tf.keras.callbacks.TensorBoard(
        log_dir="config.log_path", histogram_freq=0, write_graph=True, write_images=True
    )
    sv_loss = tf.keras.callbacks.ModelCheckpoint(
        f"{config.log_path}/{config.comment}-exp{experiment_id}-best.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )
  
    # Callback function to check transcription on the val set.
    # validation_callback = CallbackEval(model, valid_ds)
    memory_usage = MemoryUsageCallbackExtended()
    callbacks = []
    if config.save_output:
        callbacks.append(tb_logger)
        # callbacks.append(swa)
        callbacks.append(sv_loss)
    callbacks.append(memory_usage)
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    # callbacks.append(validation_callback)

    history = model.fit(
        train_ds,
        epochs=config.epochs - config.resume,
        #steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        validation_data=valid_ds,
        verbose=config.verbose,
        # validation_steps=None,
    )

    if config.save_output:  # reload the saved best weights checkpoint
        saved_based_model = f"{config.log_path}/{config.comment}-exp{experiment_id}-best.h5"
        if os.path.exists(saved_based_model):
            model.load_weights(saved_based_model)
        else:
            print(f"Warning: could not find {saved_based_model}")
    if valid_ds is not None:
        cv = model.evaluate(valid_ds, verbose=config.verbose)
    else:
        cv = None
    return model, cv, history



# In[ ]:


def train(cfg=CFG, experiment_id=0, use_supplemental=True):
    tf.keras.backend.clear_session()
    config = cfg()
    update_config_with_strategy(config)
    print(f"using {config.replicas} replicas")
    print(f"batch size {config.batch_size}")
    print(f"fp16={config.fp16}")
    seed_everything(config.seed)
    
    all_filenames = sorted(glob.glob("/kaggle/input/asl-preprocessing/records/*.tfrecord"))
    regular = [x for x in all_filenames if "supp" not in x]
    supp = [x for x in all_filenames if "supp" in x]
    
    data_filenames = regular
    if use_supplemental:
        data_filenames += supp
    print("Using TFRECORDS")
    
  
    valid_files = data_filenames[: config.num_eval]  # first part in list
    train_files = data_filenames[config.num_eval :]
    random.shuffle(train_files)
    
    
    df1 = pd.read_csv(config.input_path + "train.csv")
    df2 = pd.read_csv(config.input_path + "supplemental_metadata.csv")
    df_info = pd.concat([df1, df2])
    
    #ds = get_dataset(train_files, CFG.input_path,max_len=CFG.max_len, augment=False, batch_size=64)
    #print(ds)
    #for x,y in ds:
    #    print(x,y)
    #raise
    
    if use_supplemental:
        num_train = 3567 * 32  # with supplemental
    else:
        num_train = 1912 * 32  # without supplemental
  
    train_run(
        train_files,
        valid_files,
        config,
        num_train,
        summary=False,
        experiment_id=experiment_id,
        use_tfrecords=True,
    )
    


# In[ ]:


gc.collect()
tf.keras.backend.clear_session()


# # Train It!

# In[ ]:


#train(use_supplemental=True)


# In[ ]:


# Inference 


# In[ ]:


import tensorflow as tf


# In[ ]:


class InferModel(tf.Module):
    def __init__(self, model,config=CFG):
        super().__init__()

        self.model = model
        self.max_len=config.max_len

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None,Constants.NUM_INPUT_FEATURES), dtype=tf.float32, name="inputs")]
    )
    def __call__(self, inputs):
        """
        Applies the feature generation model and main model to the input tensor.

        Args:
            inputs: Input tensor with shape (T, F).

        Returns:
            A dictionary with a single key 'outputs' and corresponding output tensor.
        """
        x=tf.cast(inputs,tf.float32)
        x = x[None] # trick to deal with empty frames
        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, Constants.NUM_INPUT_FEATURES)), lambda: tf.identity(x))
        x = x[0]
        x = preprocess(x,max_len=self.max_len)
      
        x = self.model(x[None],training=False)[0]
                    
        x=decode_phrase(x)        
        x = tf.cond(tf.shape(x)[0] == 0, lambda: tf.zeros(1, tf.int64), lambda: tf.identity(x))                   
                    
        outputs=tf.one_hot(x,depth=59,dtype=tf.float32)
        return {"outputs": outputs}


# In[ ]:


config=CFG

model = get_model(
    max_len=config.max_len,
    output_dim=config.output_dim,
    dim=config.dim,
    input_pad=Constants.INPUT_PAD,
)
experiment_id=0

#saved_based_model = f"/kaggle/input/weights/{config.comment}-exp{experiment_id}-best.h5"
saved_based_model = f"/kaggle/input/weights-from-12h-run/{config.comment}-exp{experiment_id}-best.h5"
model.load_weights(saved_based_model)
print(f"model with weights {saved_based_model}")


# #Inference

# In[ ]:


# Sanity Check
import json
with open ("/kaggle/input/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
    character_map = json.load(f)
rev_character_map = {j:i for i,j in character_map.items()}

infer_keras_model=InferModel(model)

main_dir = '/kaggle/input/asl-fingerspelling/'
path = f'{main_dir}train_landmarks/5414471.parquet'
cols=selected_columns(path)
df = pd.read_parquet(path, engine = 'auto', columns = cols)
seq_id=1816796431
seq=df.loc[seq_id]
data = seq[cols].to_numpy()
print(f'input shape: {data.shape}, dtype: {data.dtype}')
output = infer_keras_model(data)["outputs"]
prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output, axis=1)])

print(prediction_str)


# In[ ]:


SAVED_MODEL_PATH="/kaggle/working/infer_model"

tf.saved_model.save(infer_keras_model,SAVED_MODEL_PATH)
keras_model_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
keras_model_converter.optimizations = [tf.lite.Optimize.DEFAULT]
#keras_model_converter.target_spec.supported_types = [tf.float16]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
#converter.allow_custom_ops=True
tflite_model = keras_model_converter.convert()
TFLITE_FILE_PATH="/kaggle/working/model.tflite"
with open(TFLITE_FILE_PATH, "wb") as f:
    f.write(tflite_model)

with open('/kaggle/working/inference_args.json', 'w') as f:
     json.dump({ 'selected_columns': cols }, f)



# In[ ]:


interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
REQUIRED_SIGNATURE = "serving_default"
REQUIRED_OUTPUT = "outputs"
found_signatures = list(interpreter.get_signature_list().keys())
if REQUIRED_SIGNATURE not in found_signatures:
    print("Required input signature not found.")

prediction_fn = interpreter.get_signature_runner("serving_default")
output = prediction_fn(inputs=data)
prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
print(prediction_str)


# In[ ]:


get_ipython().system('zip submission.zip "/kaggle/working/model.tflite" "/kaggle/working/inference_args.json"')


# In[ ]:


#!pip install /kaggle/input/tflite-wheels-2140/tflite_runtime_nightly-2.14.0.dev20230508-cp310-cp310-manylinux2014_x86_64.whl


# In[ ]:


"""
import json
import pandas as pd
import tflite_runtime.interpreter as tflite
import numpy as np
import time
from tqdm import tqdm
import Levenshtein as Lev
import glob
"""


# In[ ]:


"""
SEL_FEATURES = json.load(open('/kaggle/working/inference_args.json'))['selected_columns']

def load_relevant_data_subset(pq_path):
        return pd.read_parquet(pq_path, columns=SEL_FEATURES) #selected_columns)

with open ("/kaggle/input/asl-fingerspelling/character_to_prediction_index.json", "r") as f:
    character_map = json.load(f)
rev_character_map = {j:i for i,j in character_map.items()}


df = pd.read_csv('/kaggle/input/asl-fingerspelling/train.csv')

idx = 0
sample = df.loc[idx]
loaded = load_relevant_data_subset('/kaggle/input/asl-fingerspelling/' + sample['path'])
loaded = loaded[loaded.index==sample['sequence_id']].values
print(loaded.shape)
frames = loaded

def wer__(s1, s2):
    w1 = len(s1.split())
    lvd = Lev.distance(s1, s2)
    return lvd / w1

interpreter = tflite.Interpreter('model.tflite')
found_signatures = list(interpreter.get_signature_list().keys())

REQUIRED_SIGNATURE = 'serving_default'
REQUIRED_OUTPUT = 'outputs'
if REQUIRED_SIGNATURE not in found_signatures:
    raise KernelEvalException('Required input signature not found.')

prediction_fn = interpreter.get_signature_runner("serving_default")
output_lite = prediction_fn(inputs=frames)
prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output_lite[REQUIRED_OUTPUT], axis=1)])
print(prediction_str)


st = time.time()
count=0
model_time = 0

levs = []

files=glob.glob('/kaggle/input/asl-fingerspelling/train_landmarks/*.parquet')
for f in files:
    df = load_relevant_data_subset(f)
    seq=df.index.drop_duplicates()
    for ind in tqdm(seq):
        loaded = df.loc[ind].values
        count+=1
        md_st = time.time()
        output_ = prediction_fn(inputs=loaded)
        out= output_[REQUIRED_OUTPUT]
        assert out.ndim==2
        assert out.shape[1]==59
        assert out.dtype==np.float32
        assert np.all(np.isfinite(out))
        
        prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output_[REQUIRED_OUTPUT], axis=1)])
        model_time += time.time() - md_st
    
        #cur_lev = wer__(sample['phrase'], prediction_str) 
        #print(sample['phrase'], '|', prediction_str, '|', cur_lev)
        #print()

        #levs.append(cur_lev)

#print(f'WER: {np.mean(levs):.5f}')
print(f'Mean time: {(time.time() - st)/count:.2f}')
print(f'Mean time only infer: {model_time/count:.2f}')

out=prediction_fn(inputs=np.empty(0,dtype=np.float32))["outputs"]
print(out.shape,output_.dtype)
""" 

