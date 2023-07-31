import os
import gc
import numpy as np
import glob
import random
import pandas as pd
import tensorflow as tf
from .visualize import visualize_train
from .utils import (
    SWA,
    AWP,
    LevDistanceMetric,
    selected_columns,
    MemoryUsageCallbackExtended,
    seed_everything,
)
from .constants import Constants
from .config import CFG, update_config_with_strategy
from .model import get_model, CTCLoss
from .scheduler import CosineDecay


def create_gen(file_names, input_path):
    selected = selected_columns(file_names[0])
    df1 = pd.read_csv(input_path + "train.csv")
    df2 = pd.read_csv(input_path + "supplemental_metadata.csv")
    df = pd.concat([df1, df2])

    def gen():
        for file_name in file_names:
            path = "/".join(file_name.split("/")[-2:])
            seq_refs = df.loc[df.path == path]
            seqs = pd.read_parquet(file_name, columns=selected)

            for seq_id in seq_refs.sequence_id:
                coords = seqs.iloc[seqs.index == seq_id]
                if coords.empty:
                    continue
                # if np.shape(coords)[0] < 2:
                #    continue
                coords = coords.to_numpy()
                # check_nan = np.any(np.all(np.isnan(coords), axis=1))
                # if check_nan:
                #    print("nan frames", file_name, seq_id, check_nan)
                #    exit()

                phrase = str(df.loc[df.sequence_id == seq_id].phrase.iloc[0])
                label_code = [Constants.char_dict[x] for x in phrase]
                label_code = label_code
                yield coords, label_code

    return gen


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


# @tf.function()
def augment_fn(x):
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


# @tf.function()
def preprocess(x, max_len, do_pad=True):
    # shape (T,F)
    x = shrink_if_long(x, max_len=max_len)
    # Preprocess expects a batch, so we extend the dimension to (None,T,F), then reduce the output back to (T,F).
    x = tf.cast(Preprocess(max_len=max_len)(x[None, ...])[0], tf.float32)

    if do_pad:  # we can avoid this step if there is batch padding
        x = pad_if_short(x, max_len=max_len)

    return x


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

    if use_tfrecords:
        ds = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=tf.data.AUTOTUNE, compression_type="GZIP"
        )
        ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_generator(
            create_gen(filenames, input_path),
            output_signature=(
                tf.TensorSpec(
                    shape=(None, Constants.NUM_INPUT_FEATURES), dtype=tf.float32
                ),  # (T,F)
                tf.TensorSpec(shape=(None,), dtype=tf.int64),
            ),
        )
    ds.with_options(ignore_order)
    if augment:
        ds = ds.map(lambda x, y: (augment_fn(x), y), tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess(x, max_len=max_len, do_pad=False), y), tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()

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
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def explore(ds, n=10):
    counter = 0
    for feature, label in ds:  # , sequence_id in ds:
        feature = feature.numpy()  # [batch,frames,features]
        label = label.numpy()
        counter += 1
        for i in range(n):
            F = feature.shape[-1]  # number of features
            coordinates = feature[i, :, : (F // 3)].reshape(-1, F // 6, 2)
            coordinates = coordinates[:, :, :2]
            # print(coordinates)
            visualize_train("", coordinates, label[i, :])
        break


def train_run(
    train_files, valid_files, config, num_train, experiment_id=0, use_tfrecords=True, summary=False
):
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

    augment_train = True
    repeat_train = True

    shuffle_buffer = 8192
    train_ds = get_dataset(
        train_files,
        input_path=config.input_path,
        max_len=config.max_len,
        batch_size=config.batch_size,
        drop_remainder=True,
        augment=augment_train,
        repeat=repeat_train,
        shuffle_buffer=shuffle_buffer,
        use_tfrecords=use_tfrecords,
    )
    if valid_files is not None:
        valid_ds = get_dataset(
            valid_files,
            input_path=config.input_path,
            max_len=config.max_len,
            batch_size=config.batch_size,
            use_tfrecords=use_tfrecords,
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
        awp_step = config.awp_start_epoch * steps_per_epoch
        if config.awp:
            model = AWP(
                model.input, model.output, delta=config.awp_lambda, eps=0.0, start_step=awp_step
            )
        base_lr = config.lr
        # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        lr_schedule = CosineDecay(
            initial_learning_rate=base_lr / 10,
            decay_steps=int(0.95 * steps_per_epoch * config.epochs),
            alpha=0.02,
            name=None,
            warmup_target=base_lr,
            warmup_steps=int(0.05 * steps_per_epoch * config.epochs),
        )

        opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=config.weight_decay)
        # opt = tf.keras.optimizers.AdamW(learning_rate=base_lr, weight_decay=config.weight_decay)
        loss = CTCLoss(pad_token_idx=Constants.LABEL_PAD)

        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=[
                LevDistanceMetric(),
            ],
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
    memory_usage = MemoryUsageCallbackExtended()
    # stochastic weight averaging
    # swa = SWA(
    #    f"{config.log_path}/{config.comment}-exp{experiment_id}",
    #    config.swa_epochs,
    #    strategy=strategy,
    #    train_ds=train_ds,
    #    valid_ds=valid_ds,
    # )

    # Callback function to check transcription on the val set.
    # validation_callback = CallbackEval(model, valid_ds)
    callbacks = []
    if config.save_output:
        callbacks.append(tb_logger)
        # callbacks.append(swa)
        callbacks.append(sv_loss)

    callbacks.append(memory_usage)
    # callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    # callbacks.append(validation_callback)

    history = model.fit(
        train_ds,
        epochs=config.epochs - config.resume,
        steps_per_epoch=steps_per_epoch,
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


def train(cfg=CFG, experiment_id=0, use_supplemental=True, use_tfrecords=False):
    tf.keras.backend.clear_session()
    config = cfg()
    update_config_with_strategy(config)
    print(f"using {config.replicas} replicas")
    seed_everything(config.seed)

    if use_tfrecords:
        all_filenames = sorted(glob.glob("/kaggle/input/sign-tfrecords/*.tfrecord"))
        regular = [x for x in all_filenames if "train" not in x]
        supp = [x for x in all_filenames if "supp" in x]
        chicago = [x for x in all_filenames if "chicago" in x]
        # data_filenames = regular
        # if use_supplemental:
        #    data_filenames += supp
        data_filenames = chicago
        print("Using TFRECORDS")
    else:
        # data_filenames = sorted(glob.glob(config.input_path + "train_landmarks/*.parquet"))
        data_filenames = sorted(glob.glob(config.input_path + "chicago/*.parquet"))
        if use_supplemental:
            data_filenames += sorted(
                glob.glob(config.input_path + "supplemental_landmarks/*.parquet")
            )
        print("Using Parquet")

    ds = get_dataset(
        data_filenames,
        input_path=config.input_path,
        max_len=CFG.max_len,
        augment=True,
        batch_size=64,
        use_tfrecords=use_tfrecords,
    )

    # for x, y in ds:
    #   print(x.shape, y.shape)

    explore(ds)
    exit()

    valid_files = data_filenames[: config.num_eval]  # first part in list
    train_files = data_filenames[config.num_eval :]
    random.shuffle(train_files)

    if use_supplemental:
        num_train = 3567 * 32  # with supplemental
    else:
        num_train = 1912 * 32  # without supplemental

    print(num_train)
    train_run(
        train_files,
        valid_files,
        config,
        num_train,
        summary=True,
        experiment_id=experiment_id,
        use_tfrecords=use_tfrecords,
    )
