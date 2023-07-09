import os
from matplotlib import pyplot as plt
import numpy as np
from Levenshtein import distance as Lev_distance
from config import CFG
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402


def get_char_dict():
    char_dict_file = f"{CFG.input_path}/character_to_prediction_index.json"
    with open(char_dict_file) as f:
        char_dict = json.load(f)
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
    LIP = [
        0,
        61,
        185,
        40,
        39,
        37,
        267,
        269,
        270,
        409,
        291,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        78,
        191,
        80,
        81,
        82,
        13,
        312,
        311,
        310,
        415,
        95,
        88,
        178,
        87,
        14,
        317,
        402,
        318,
        324,
        308,
    ]

    REYE = [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        246,
        161,
        160,
        159,
        158,
        157,
        173,
    ]
    LEYE = [
        263,
        249,
        390,
        373,
        374,
        380,
        381,
        382,
        362,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]

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

    POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
    # POINT_LANDMARKS = list(range(543))

    NUM_NODES = len(POINT_LANDMARKS)
    CHANNELS = 6 * NUM_NODES


class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Unified single-cycle learning rate scheduler for tensorflow.
    2022 Hoyeol Sohn <hoeyol0730@gmail.com>
    """

    def __init__(
        self,
        lr=1e-4,
        epochs=10,
        steps_per_epoch=100,
        steps_per_update=1,
        resume_epoch=0,
        decay_epochs=10,
        sustain_epochs=0,
        warmup_epochs=0,
        lr_start=0,
        lr_min=0,
        warmup_type="linear",
        decay_type="cosine",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr = float(lr)
        self.epochs = float(epochs)
        self.steps_per_update = float(steps_per_update)
        self.resume_epoch = float(resume_epoch)
        self.steps_per_epoch = float(steps_per_epoch)
        self.decay_epochs = float(decay_epochs)
        self.sustain_epochs = float(sustain_epochs)
        self.warmup_epochs = float(warmup_epochs)
        self.lr_start = float(lr_start)
        self.lr_min = float(lr_min)
        self.decay_type = decay_type
        self.warmup_type = warmup_type

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # total_steps = self.epochs * self.steps_per_epoch
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        sustain_steps = self.sustain_epochs * self.steps_per_epoch
        decay_steps = self.decay_epochs * self.steps_per_epoch

        if self.resume_epoch > 0:
            step = step + self.resume_epoch * self.steps_per_epoch

        step = tf.cond(step > decay_steps, lambda: decay_steps, lambda: step)
        step = tf.math.truediv(step, self.steps_per_update) * self.steps_per_update

        warmup_cond = step < warmup_steps
        decay_cond = step >= (warmup_steps + sustain_steps)

        if self.warmup_type == "linear":
            lr = tf.cond(
                warmup_cond,
                lambda: tf.math.divide_no_nan(self.lr - self.lr_start, warmup_steps) * step
                + self.lr_start,
                lambda: self.lr,
            )
        elif self.warmup_type == "exponential":
            factor = tf.pow(self.lr_start, 1 / warmup_steps)
            lr = tf.cond(
                warmup_cond,
                lambda: (self.lr - self.lr_start) * factor ** (warmup_steps - step) + self.lr_start,
                lambda: self.lr,
            )
        elif self.warmup_type == "cosine":
            lr = tf.cond(
                warmup_cond,
                lambda: 0.5
                * (self.lr - self.lr_start)
                * (1 + tf.cos(3.14159265359 * (warmup_steps - step) / warmup_steps))
                + self.lr_start,
                lambda: self.lr,
            )
        else:
            raise NotImplementedError

        if self.decay_type == "linear":
            lr = tf.cond(
                decay_cond,
                lambda: self.lr
                + (self.lr_min - self.lr)
                / (decay_steps - warmup_steps - sustain_steps)
                * (step - warmup_steps - sustain_steps),
                lambda: lr,
            )
        elif self.decay_type == "exponential":
            factor = tf.pow(self.lr_min, 1 / (decay_steps - warmup_steps - sustain_steps))
            lr = tf.cond(
                decay_cond,
                lambda: (self.lr - self.lr_min) * factor ** (step - warmup_steps - sustain_steps)
                + self.lr_min,
                lambda: lr,
            )
        elif self.decay_type == "cosine":
            lr = tf.cond(
                decay_cond,
                lambda: 0.5
                * (self.lr - self.lr_min)
                * (
                    1
                    + tf.cos(
                        3.14159265359
                        * (step - warmup_steps - sustain_steps)
                        / (decay_steps - warmup_steps - sustain_steps)
                    )
                )
                + self.lr_min,
                lambda: lr,
            )
        else:
            raise NotImplementedError

        return lr

    def plot(self):
        step = max(
            1, int(self.epochs * self.steps_per_epoch) // 1000
        )  # 1 for total_steps < 1000, total_steps//1000 else
        eps = list(range(0, int(self.epochs * self.steps_per_epoch), step))
        learning_rates = [self(x) for x in eps]
        plt.scatter(eps, learning_rates, 2)
        plt.show()


class ListedLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, schedules, steps_per_epoch=100, update_per_epoch=1, **kwargs):
        super().__init__(**kwargs)
        self.schedules = schedules
        self.steps_per_epoch = float(steps_per_epoch)
        self.update_per_epoch = float(update_per_epoch)
        for s in self.schedules:
            s.steps_per_epoch = float(steps_per_epoch)
            s.update_per_epoch = float(update_per_epoch)
        self.restart_epochs = tf.math.cumsum([s.epochs for s in self.schedules])
        self.epochs = self.restart_epochs[-1]
        self.global_steps = tf.math.cumsum([s.epochs * s.steps_per_epoch for s in self.schedules])

        # print(self.fns)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        idx = tf.searchsorted(self.global_steps, [step + 1])[0]
        global_steps = tf.concat([[0], self.global_steps], 0)
        # fns = [lambda: self.schedules[i].__call__(step-global_steps[i]) for i in range(len(self.schedules))]
        fns = [
            (lambda x: (lambda: self.schedules[x].__call__(step - global_steps[x])))(i)
            for i in range(len(self.schedules))
        ]
        r = tf.switch_case(idx, branch_fns=fns)
        return r

    def plot(self):
        step = max(
            1, int(self.epochs * self.steps_per_epoch) // 1000
        )  # 1 for total_steps < 1000, total_steps//1000 else
        eps = list(range(0, int(self.epochs * self.steps_per_epoch), step))
        learning_rates = [self(x) for x in eps]
        plt.scatter(eps, learning_rates, 2)
        plt.show()


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
        valid_steps=None,
    ):
        super().__init__()
        self.swa_epochs = swa_epochs
        self.swa_weights = None
        self.save_name = save_name
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.train_steps = train_steps
        self.valid_steps = valid_steps
        self.strategy = strategy

    @tf.function(jit_compile=True)
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
                self.model.evaluate(self.valid_ds, steps=self.valid_steps)


class FGM(tf.keras.Model):
    # Fast Gradient Method
    def __init__(self, *args, delta=0.2, eps=1e-4, start_step=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.eps = eps
        self.start_step = start_step

    def train_step_fgm(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        embedding = self.trainable_variables[0]
        embedding_gradients = tape.gradient(loss, [self.trainable_variables[0]])[0]
        embedding_gradients = tf.zeros_like(embedding) + embedding_gradients
        delta = tf.math.divide_no_nan(
            self.delta * embedding_gradients,
            tf.math.sqrt(tf.reduce_sum(embedding_gradients**2)) + self.eps,
        )
        self.trainable_variables[0].assign_add(delta)
        with tf.GradientTape() as tape2:
            y_pred = self(x, training=True)
            new_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            if hasattr(self.optimizer, "get_scaled_loss"):
                new_loss = self.optimizer.get_scaled_loss(new_loss)
        gradients = tape2.gradient(new_loss, self.trainable_variables)
        if hasattr(self.optimizer, "get_unscaled_gradients"):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.trainable_variables[0].assign_sub(delta)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # self_loss.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return tf.cond(
            self._train_counter < self.start_step,
            lambda: super(FGM, self).train_step(data),
            lambda: self.train_step_fgm(data),
        )


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


class Snapshot(tf.keras.callbacks.Callback):
    def __init__(self, save_name, snapshot_epochs=[]):
        super().__init__()
        self.snapshot_epochs = snapshot_epochs
        self.save_name = save_name

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        if epoch in self.snapshot_epochs:  # your custom condition
            self.model.save_weights(f"{self.save_name}-epoch{epoch}.h5")
        self.model.save_weights(f"{self.save_name}-last.h5")


def num_to_char_fn(y):
    return [Constants.inv_dict.get(x, "") for x in y]


@tf.function(jit_compile=True)
def decode_phrase(pred):
    pad_token_idx = 59
    x = tf.argmax(pred, axis=1)
    diff = tf.not_equal(x[:-1], x[1:])
    adjacent_indices = tf.where(diff)[:, 0]
    x = tf.gather(x, adjacent_indices)
    mask = x != pad_token_idx
    x = tf.boolean_mask(x, mask, axis=0)
    return x


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    output_text = []
    for result in pred:
        result = "".join(num_to_char_fn(decode_phrase(result).numpy()))
        output_text.append(result)
    return output_text


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


def metric(val_ds, model):
    global_N, global_D = 0, 0
    count = 0
    for batch in val_ds:
        count += 1
        print(count)
        feature, label = batch
        logits = model(feature)
        label = label.numpy()
        target_strings = convert_to_strings(label)
        predict_strings = decode_batch_predictions(logits)
        values = [
            calculate_N_D(target, predict)
            for target, predict in zip(target_strings, predict_strings)
        ]
        global_D += np.sum([x[0] for x in values])
        global_N += np.sum([x[1] for x in values])

    metric_value = np.clip((global_N - global_D) / global_N, a_min=0, a_max=1)
    return metric_value
