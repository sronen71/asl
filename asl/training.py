import os
import gc
import numpy as np
import glob
import random
import pandas as pd
from .visualize import visualize_train
from .utils import Snapshot, SWA, FGM, AWP
from .constants import Constants
from .config import CFG
from .model import get_model, CTCLoss

import tensorflow as tf


def selected_columns():
    file_example = glob.glob(CFG.input_path + "train_landmarks/*.parquet")[0]
    df = pd.read_parquet(file_example)
    selected_x = df.columns[[x + 1 for x in Constants.POINT_LANDMARKS]].tolist()
    selected_y = [c.replace("x", "y") for c in selected_x]
    selected_z = [c.replace("x", "z") for c in selected_x]
    selected = []
    for i in range(Constants.NUM_NODES):
        selected.append(selected_x[i])
        selected.append(selected_y[i])
        selected.append(selected_z[i])
    return selected


def create_gen(file_names):
    selected = selected_columns()
    df1 = pd.read_csv(CFG.input_path + "train.csv")
    df2 = pd.read_csv(CFG.input_path + "supplemental_metadata.csv")
    df = pd.concat([df1, df2])

    def gen():
        for file_name in file_names:
            path = "/".join(file_name.split("/")[-2:])
            seq_refs = df.loc[df.path == path]
            seqs = pd.read_parquet(file_name, columns=selected)

            for seq_id in seq_refs.sequence_id:
                coords = seqs.iloc[seqs.index == seq_id].to_numpy()
                if coords.shape[0] < 2:
                    continue
                coords = coords.reshape((coords.shape[0], -1, 3))
                phrase = str(df.loc[df.sequence_id == seq_id].phrase.iloc[0])
                label = [Constants.char_dict[x] for x in phrase]
                out = {"coordinates": coords, "label": label}
                yield out

    return gen


def resize_pad(x, max_len):
    # shape T,F,3
    if tf.shape(x)[0] < max_len:
        x = tf.pad(
            x,
            ([[0, max_len - tf.shape(x)[0]], [0, 0], [0, 0]]),
            constant_values=float("nan"),
        )
    else:
        x = tf.image.resize(x, (max_len, tf.shape(x)[1]))
    return x


def count_data_items(dataset):
    dataset_size = 0
    for _ in dataset:
        dataset_size += 1
        print(dataset_size)
    return dataset_size


# Seed all random number generators
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len, format="parquet", **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        if format == "parquet":
            self.landmarks = None
            self.center = Constants.CENTER_INDICES
        else:
            self.landmarks = Constants.POINT_LANDMARKS
            self.center = Constants.CENTER_LANDMARKS  # 17

    def call(self, inputs):
        if tf.rank(inputs) == 3:
            x = inputs[None, ...]
        else:
            x = inputs

        mean = tf_nan_mean(tf.gather(x, self.center, axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        if self.landmarks is not None:
            x = tf.gather(x, self.landmarks, axis=2)  # N,T,P,C
        std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)

        x = (x - mean) / std

        if self.max_len is not None:
            x = x[:, : self.max_len]
        x = x[..., :2]

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
                tf.reshape(x, (-1, length, 2 * Constants.NUM_NODES)),
                tf.reshape(dx, (-1, length, 2 * Constants.NUM_NODES)),
                tf.reshape(dx2, (-1, length, 2 * Constants.NUM_NODES)),
            ],
            axis=-1,
        )

        x = tf.where(tf.math.is_nan(x), tf.constant(0.0, x.dtype), x)
        return x


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

    x, y, z = tf.unstack(x, axis=-1)
    x = 1 - x
    new_x = tf.stack([x, y, z], -1)
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
        tf.fill([mask_size, tf.shape(x)[1], 3], mask_value),
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


def augment_fn(x, always=False, max_len=None):
    if tf.random.uniform(()) < 0.4 or always:
        x = resample(x, (0.5, 1.5))
    if tf.random.uniform(()) < 0.4 or always:
        x = flip_lr(x)
    if tf.random.uniform(()) < 0.4 or always:
        x = spatial_random_affine(x)
    if tf.random.uniform(()) < 0.2 or always:
        x = temporal_mask(x)
    if tf.random.uniform(()) < 0.2 or always:
        x = spatial_mask(x)
    return x


def filter_nans_tf(x, ref_points=Constants.POINT_LANDMARKS):
    mask = tf.math.logical_not(
        tf.reduce_all(tf.math.is_nan(tf.gather(x, ref_points, axis=1)), axis=[-2, -1])
    )
    x = tf.boolean_mask(x, mask, axis=0)
    return x


def decode_tfrec(record_bytes):
    features = tf.io.parse_single_example(
        record_bytes,
        {
            "coordinates": tf.io.VarLenFeature(tf.float32),
            "label": tf.io.VarLenFeature(tf.int64),
            "sequence_id": tf.io.FixedLenFeature(1, tf.int64),
        },
    )
    out = {}
    out["coordinates"] = tf.reshape(
        tf.sparse.to_dense(features["coordinates"]), (-1, Constants.ROWS_PER_FRAME, 3)
    )
    out["label"] = tf.sparse.to_dense(features["label"])
    return out


def preprocess(x, max_len, augment=False, format="parquet"):
    coord = x["coordinates"]
    if format == "parquet":
        ref_points = list(range(Constants.NUM_NODES))
    else:
        ref_points = Constants.POINT_LANDMARKS
    coord = filter_nans_tf(coord, ref_points)

    if augment:
        coord = augment_fn(coord, max_len=max_len)
    coord = resize_pad(coord, max_len=max_len)

    if format == "parquet":
        nrows = Constants.NUM_NODES
    else:
        nrows = Constants.ROWS_PER_FRAME
    coord = tf.ensure_shape(coord, (None, nrows, 3))
    coord = tf.cast(Preprocess(max_len=max_len, format=format)(coord)[0], tf.float32)

    return (coord, x["label"])


def get_dataset(
    filenames,
    max_len,
    batch_size=64,
    drop_remainder=False,
    augment=False,
    shuffle=False,
    repeat=False,
):
    if "tfrecord" in filenames[0]:
        format = "tfrecord"
        print("TFRECORD dataset")
        ds = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=tf.data.AUTOTUNE, compression_type="GZIP"
        )
        ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    else:
        format = "parquet"
        print("Generator Dataset")
        ds = tf.data.Dataset.from_generator(
            create_gen(filenames),
            output_signature={
                "coordinates": tf.TensorSpec(
                    shape=(None, Constants.NUM_NODES, 3), dtype=tf.float32
                ),
                "label": tf.TensorSpec(shape=(None,), dtype=tf.int64),
            },
        )
        ds = ds.cache()
    ds = ds.map(
        lambda x: preprocess(x, augment=augment, max_len=max_len, format=format), tf.data.AUTOTUNE
    )
    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(shuffle)
        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)

    if batch_size:
        ds = ds.padded_batch(
            batch_size,
            padding_values=(
                tf.constant(Constants.INPUT_PAD, dtype=tf.float32),
                tf.constant(Constants.LABEL_PAD, dtype=tf.int64),
            ),
            padded_shapes=([max_len, Constants.CHANNELS], [Constants.MAX_STRING_LEN]),
            drop_remainder=drop_remainder,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def explore(ds):
    counter = 0
    for feature, label in ds:  # , sequence_id in ds:
        feature = feature.numpy()
        label = label.numpy()
        counter += 1
        for i in range(10):
            N = feature.shape[-1]
            coordinates = feature[i, :].reshape(-1, N // 6, 6)
            coordinates = coordinates[:, :, :2]
            visualize_train("", coordinates, label[i, :])
        break


def train_run(config, train_files, valid_files=None, summary=True, fold=0):
    seed_everything(config.seed)
    tf.keras.backend.clear_session()
    gc.collect()
    tf.config.optimizer.set_jit(True)

    if config.fp16:
        # policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
        # tf.keras.mixed_precision.set_global_policy(policy)
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        policy = tf.keras.mixed_precision.Policy("float32")
        tf.keras.mixed_precision.set_global_policy(policy)
    augment_train = True
    repeat_train = True
    # augment_train = False
    # repeat_train = False

    shuffle = 8192
    train_ds = get_dataset(
        train_files,
        max_len=config.max_len,
        batch_size=config.batch_size,
        drop_remainder=True,
        augment=augment_train,
        repeat=repeat_train,
        shuffle=shuffle,
    )
    if valid_files is not None:
        valid_ds = get_dataset(
            valid_files,
            batch_size=config.batch_size,
            max_len=config.max_len,
        )
    else:
        valid_ds = None
        valid_files = []

    # num_train = count_data_items(train_ds)
    # num_valid = count_data_items(valid_ds)
    # print(num_train, num_valid, config.batch_size)
    num_train = 1716 * 32  # without supplemental
    num_valid = 191 * 32  # 10%
    # num_train = 3401 * 32  # with supplemental
    # num_valid = 352 * 32  # 10%
    steps_per_epoch = num_train // config.batch_size
    valid_steps = num_valid // config.batch_size
    strategy = config.strategy
    with strategy.scope():
        model = get_model(
            max_len=config.max_len,
            output_dim=config.output_dim,
            input_pad=Constants.INPUT_PAD,
            dim=config.dim,
        )
        awp_step = config.awp_start_epoch * steps_per_epoch
        if config.fgm:
            model = FGM(
                model.input, model.output, delta=config.awp_lambda, eps=0.0, start_step=awp_step
            )
        elif config.awp:
            model = AWP(
                model.input, model.output, delta=config.awp_lambda, eps=0.0, start_step=awp_step
            )
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.lr / 10,
            decay_steps=int(0.95 * steps_per_epoch),
            alpha=0.01,
            name=None,
            warmup_target=config.lr,
            warmup_steps=int(0.05 * steps_per_epoch),
        )

        opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=config.weight_decay)
        # opt = tf.keras.optimizers.AdamW(learning_rate=config.lr, weight_decay=config.weight_decay)
        # opt = RectifiedAdam(
        #    learning_rate=schedule, weight_decay=decay_schedule, sma_threshold=4
        # )
        # opt = Lookahead(opt, sync_period=5)
        loss = CTCLoss(pad_token_idx=Constants.LABEL_PAD)

        model.compile(
            optimizer=opt,
            loss=loss,
            # metrics=[
            #    [
            #        tf.keras.metrics.CategoricalAccuracy(),
            #    ],
            # ],
        )

    if summary:
        print()
        model.summary()
        print()
        print(train_ds, valid_ds)
        print()
        # schedule.plot()
        # print()
    print(f"---------fold{fold}---------")
    print(f"train:{num_train} valid:{num_valid}")
    print()

    if config.resume:
        print(f"resume from epoch{config.resume}")
        model.load_weights(f"{config.log_path}/{config.comment}-fold{fold}-last.h5")
        if train_ds is not None:
            model.evaluate(train_ds.take(steps_per_epoch))
        if valid_ds is not None:
            model.evaluate(valid_ds)

    csv_logger = tf.keras.callbacks.CSVLogger(
        f"{config.log_path}/{config.comment}-fold{fold}-logs.csv"
    )
    tb_logger = tf.keras.callbacks.TensorBoard(
        log_dir="config.log_path", histogram_freq=0, write_graph=True, write_images=True
    )
    sv_loss = tf.keras.callbacks.ModelCheckpoint(
        f"{config.log_path}/{config.comment}-fold{fold}-best.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )
    snap = Snapshot(f"{config.log_path}/{config.comment}-fold{fold}", config.snapshot_epochs)
    # stochastic weight averaging
    swa = SWA(
        f"{config.log_path}/{config.comment}-fold{fold}",
        config.swa_epochs,
        strategy=strategy,
        train_ds=train_ds,
        valid_ds=valid_ds,
        valid_steps=valid_steps,
    )

    # Callback function to check transcription on the val set.
    # validation_callback = CallbackEval(model, valid_ds)
    callbacks = []
    if config.save_output:
        callbacks.append(csv_logger)
        callbacks.append(tb_logger)
        callbacks.append(snap)
        # callbacks.append(swa)
        # if fold != "all":
        callbacks.append(sv_loss)

    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    # callbacks.append(validation_callback)

    history = model.fit(
        train_ds,
        epochs=config.epoch - config.resume,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        validation_data=valid_ds,
        verbose=config.verbose,
        # validation_steps=None,
    )

    if config.save_output:  # reload the saved best weights checkpoint
        saved_based_model = f"{config.log_path}/{config.comment}-fold{fold}-best.h5"
        if os.path.exists(saved_based_model):
            model.load_weights(saved_based_model)
        else:
            print(f"Warning: could not find {saved_based_model}")
    if fold != "all":
        cv = model.evaluate(valid_ds, verbose=config.verbose, steps=valid_steps)
    else:
        cv = None

    return model, cv, history


def train_eval(filenames, config=CFG, summary=True):
    n = len(filenames)
    num_val = int(0.1 * n)
    valid_files = filenames[:num_val]
    train_files = filenames[num_val:]

    train_run(config, train_files, valid_files, summary=True)


def train(config=CFG, format="parquet"):
    format = "tfrecord"
    tf.keras.backend.clear_session()
    if format == "parquet":
        data_filenames1 = sorted(glob.glob(config.input_path + "train_landmarks/*.parquet"))
        data_filenames2 = sorted(glob.glob(config.input_path + "supplemental_landmarks/*.parquet"))
    else:
        data_filenames1 = sorted(glob.glob(config.output_path + "records/*.tfrecord"))
        data_filenames2 = sorted(glob.glob(config.output_path + "sup_records/*.tfrecord"))

    # data_filenames = data_filenames1 + data_filenames2
    data_filenames = data_filenames1

    # gen = create_gen(data_filenames[:2])
    # x = next(iter(gen()))
    # print(x)
    # exit()

    # ds = get_dataset(data_filenames, max_len=CFG.max_len, augment=True, batch_size=1024)
    # explore(ds)
    # exit()

    train_eval(data_filenames)
