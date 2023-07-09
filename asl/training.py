import os
import gc
import numpy as np
import glob
import random
from .visualize import visualize_train
from .utils import OneCycleLR, Snapshot, SWA, FGM, AWP, Constants
from .config import CFG
from .model import get_model, CTCLoss

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402
import tensorflow_addons as tfa  # noqa: E402


# tf.autograph.set_verbosity(10, alsologtostdout=True)


def resize_pad(x, max_len):
    if tf.shape(x)[1] < max_len:
        x = tf.pad(
            x,
            ([[0, max_len - tf.shape(x)[0]], [0, 0], [0, 0]]),
            constant_values=Constants.INPUT_PAD,
        )
    else:
        x = tf.image.resize(x, (max_len, tf.shape(x)[1]))
    return x


def count_data_items(dataset):
    dataset_size = 0
    for _ in dataset:
        dataset_size += 1
    return dataset_size


# Seed all random number generators
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_strategy(device="GPU"):
    is_tpu = False
    if "TPU" in device:
        tpu = "local" if device == "TPU-VM" else None
        print("connecting to TPU...")
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        is_tpu = True

    if device == "GPU":
        ngpu = len(tf.config.experimental.list_physical_devices("GPU"))
        if ngpu > 1:
            print("Using multi GPU")
            strategy = tf.distribute.MirroredStrategy()
        elif ngpu == 1:
            print("Using single GPU")
            strategy = tf.distribute.get_strategy()

    if device == "GPU":
        print("Num GPUs Available: ", ngpu)

    REPLICAS = strategy.num_replicas_in_sync
    print(f"REPLICAS: {REPLICAS}")

    return strategy, REPLICAS, is_tpu


STRATEGY, N_REPLICAS, IS_TPU = get_strategy()


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
    def __init__(self, max_len, landmarks=Constants.POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.landmarks = Constants.POINT_LANDMARKS

    def call(self, inputs):
        if tf.rank(inputs) == 3:
            x = inputs[None, ...]
        else:
            x = inputs

        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x = tf.gather(x, self.landmarks, axis=2)  # N,T,P,C
        std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)

        x = (x - mean) / std
        if self.max_len is not None:
            x = x[:, : self.max_len]
        length = tf.shape(x)[1]
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
        x = tf.concat(
            [
                tf.reshape(x, (-1, length, 2 * len(self.landmarks))),
                tf.reshape(dx, (-1, length, 2 * len(self.landmarks))),
                tf.reshape(dx2, (-1, length, 2 * len(self.landmarks))),
            ],
            axis=-1,
        )
        # x = tf.reshape(x, (-1, length, 2 * len(self.Constants.POINT_LANDMARKS)))

        x = tf.where(tf.math.is_nan(x), tf.constant(0.0, x.dtype), x)
        return x


def flip_lr(x):
    x, y, z = tf.unstack(x, axis=-1)
    x = 1 - x
    new_x = tf.stack([x, y, z], -1)
    new_x = tf.transpose(new_x, [1, 0, 2])
    lhand = tf.gather(new_x, Constants.LHAND, axis=0)
    rhand = tf.gather(new_x, Constants.RHAND, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(Constants.LHAND)[..., None], rhand)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(Constants.RHAND)[..., None], lhand)
    llip = tf.gather(new_x, Constants.LLIP, axis=0)
    rlip = tf.gather(new_x, Constants.RLIP, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(Constants.LLIP)[..., None], rlip)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(Constants.RLIP)[..., None], llip)
    lpose = tf.gather(new_x, Constants.LPOSE, axis=0)
    rpose = tf.gather(new_x, Constants.RPOSE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(Constants.LPOSE)[..., None], rpose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(Constants.RPOSE)[..., None], lpose)
    leye = tf.gather(new_x, Constants.LEYE, axis=0)
    reye = tf.gather(new_x, Constants.REYE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(Constants.LEYE)[..., None], reye)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(Constants.REYE)[..., None], leye)
    lnose = tf.gather(new_x, Constants.LNOSE, axis=0)
    rnose = tf.gather(new_x, Constants.RNOSE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(Constants.LNOSE)[..., None], rnose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(Constants.RNOSE)[..., None], lnose)
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


def temporal_crop(x, max_length):
    # crop randomly if number of frames greater than maximum length
    l0 = tf.shape(x)[0]
    offset = tf.random.uniform(
        (), 0, tf.clip_by_value(l0 - max_length, 1, max_length), dtype=tf.int32
    )
    x = x[offset : (offset + max_length)]
    return x


def temporal_mask(x, size=(0.05, 0.1), mask_value=float("nan")):
    l0 = tf.shape(x)[0]
    mask_size = tf.random.uniform((), *size)
    mask_size = tf.cast(tf.cast(l0, tf.float32) * mask_size, tf.int32)
    mask_offset = tf.random.uniform((), 0, tf.clip_by_value(l0 - mask_size, 1, l0), dtype=tf.int32)
    x = tf.tensor_scatter_nd_update(
        x,
        tf.range(mask_offset, mask_offset + mask_size)[..., None],
        tf.fill([mask_size, 543, 3], mask_value),
    )
    return x


def spatial_mask(x, size=(0.1, 0.3), mask_value=float("nan")):
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
    # if max_len is not None:
    #    x = temporal_crop(x, max_len)
    if tf.random.uniform(()) < 0.4 or always:
        x = spatial_random_affine(x)
    if tf.random.uniform(()) < 0.2 or always:
        x = temporal_mask(x)
    if tf.random.uniform(()) < 0.2 or always:
        x = spatial_mask(x)
    return x


def filter_nans_tf(x, ref_point=Constants.POINT_LANDMARKS):
    mask = tf.math.logical_not(
        tf.reduce_all(tf.math.is_nan(tf.gather(x, ref_point, axis=1)), axis=[-2, -1])
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
    # out["sequence_id"] = features["sequence_id"][0]
    return out


def preprocess(x, max_len, augment=False):
    coord = x["coordinates"]

    coord = filter_nans_tf(coord)
    if augment:
        coord = augment_fn(coord, max_len=max_len)
    coord = resize_pad(coord, max_len=max_len)
    coord = tf.ensure_shape(coord, (None, Constants.ROWS_PER_FRAME, 3))
    return (
        tf.cast(Preprocess(max_len=max_len)(coord)[0], tf.float32),
        x["label"],
    )  # x["sequence_id"]


def get_tfrec_dataset(
    tfrecords,
    max_len,
    batch_size=64,
    drop_remainder=False,
    augment=False,
    shuffle=False,
    repeat=False,
):
    # Initialize dataset with TFRecords
    ds = tf.data.TFRecordDataset(
        tfrecords, num_parallel_reads=tf.data.AUTOTUNE, compression_type="GZIP"
    )
    ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    ds = ds.map(lambda x: preprocess(x, augment=augment, max_len=max_len), tf.data.AUTOTUNE)

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
                # tf.constant(0, dtype=tf.int64),
            ),
            padded_shapes=([max_len, Constants.CHANNELS], [Constants.MAX_STRING_LEN]),
            drop_remainder=drop_remainder,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def explore(ds):
    # char_dict = get_char_dict()
    # inv_dict = {v: k for k, v in char_dict.items()}
    counter = 0
    for feature, label in ds:  # , sequence_id in ds:
        feature = feature.numpy()
        label = label.numpy()
        # sequence_id = sequence_id.numpy()
        # label_str = "".join([inv_dict[x] for x in label[0, :].tolist()])
        # print(
        #    counter,
        #    len(sequence_id),
        #    feature.shape,
        #    label.shape,
        #    np.min(feature),
        #    np.max(feature),
        #    label_str,
        # )
        counter += 1
        # print(sequence_id, feature.shape, label.shape)
        for i in range(3):
            N = feature.shape[-1]
            coordinates = feature[i, :].reshape(-1, N // 6, 6)
            coordinates = coordinates[:, :, :2]
            # visualize_train(sequence_id[i], coordinates, label[i, :])
            visualize_train("", coordinates, label[i, :])
        break


def train_fold(config, fold, train_files, valid_files=None, strategy=STRATEGY, summary=True):
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
    shuffle = 16384
    if fold != "all":
        train_ds = get_tfrec_dataset(
            train_files,
            max_len=config.max_len,
            batch_size=config.batch_size,
            drop_remainder=True,
            augment=augment_train,
            repeat=repeat_train,
            shuffle=shuffle,
        )
        valid_ds = get_tfrec_dataset(
            valid_files,
            batch_size=config.batch_size,
            max_len=config.max_len,
        )
    else:
        train_ds = get_tfrec_dataset(
            train_files,
            batch_size=config.batch_size,
            max_len=config.max_len,
            drop_remainder=False,
            augment=augment_train,
            repeat=repeat_train,
            shuffle=shuffle,
        )
        valid_ds = None
        valid_files = []

    # num_train = count_data_items(train_ds)
    # num_valid = count_data_items(valid_ds)
    num_train = 1716 * 32
    num_valid = 383 * 32
    steps_per_epoch = num_train // config.batch_size
    valid_steps = num_valid // config.batch_size
    with strategy.scope():
        dropout_step = config.dropout_start_epoch * steps_per_epoch
        # model = get_model(
        #    max_len=config.max_len,
        #    dropout_step=dropout_step,
        #    dim=config.dim,
        #    input_pad=INPUT_PAD,
        #    output_dim=config.output_dim,
        #
        # )
        model = get_model(
            max_len=config.max_len,
            output_dim=config.output_dim,
            input_pad=Constants.INPUT_PAD,
            dim=config.dim,
        )

        schedule = OneCycleLR(
            lr=config.lr,
            epochs=config.epoch,
            warmup_epochs=config.epoch * config.warmup,
            steps_per_epoch=steps_per_epoch,
            resume_epoch=config.resume,
            decay_epochs=config.epoch,
            lr_min=config.lr_min,
            decay_type=config.decay_type,
            warmup_type="linear",
        )
        decay_schedule = OneCycleLR(
            config.lr * config.weight_decay,
            config.epoch,
            warmup_epochs=config.epoch * config.warmup,
            steps_per_epoch=steps_per_epoch,
            resume_epoch=config.resume,
            decay_epochs=config.epoch,
            lr_min=config.lr_min * config.weight_decay,
            decay_type=config.decay_type,
            warmup_type="linear",
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

        opt = tfa.optimizers.RectifiedAdam(
            learning_rate=schedule, weight_decay=decay_schedule, sma_threshold=4
        )  # , clipvalue=1.)
        opt = tfa.optimizers.Lookahead(opt, sync_period=5)

        # loss=[
        #    tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
        # ]
        loss = CTCLoss(pad_token_idx=Constants.LABEL_PAD)
        # loss = CTCLoss3

        model.compile(
            optimizer=opt,
            loss=loss,
            # metrics=[
            #    [
            #        tf.keras.metrics.CategoricalAccuracy(),
            #    ],
            # ],
            # steps_per_execution=10,
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
        model.load_weights(f"{config.output_dir}/{config.comment}-fold{fold}-last.h5")
        if train_ds is not None:
            model.evaluate(train_ds.take(steps_per_epoch))
        if valid_ds is not None:
            model.evaluate(valid_ds)

    # logger = tf.keras.callbacks.CSVLogger(
    #    f"{config.output_dir}/{config.comment}-fold{fold}-logs.csv"
    # )
    logger = tf.keras.callbacks.TensorBoard(
        log_dir="config.output_dir", histogram_freq=0, write_graph=True, write_images=True
    )
    sv_loss = tf.keras.callbacks.ModelCheckpoint(
        f"{config.output_dir}/{config.comment}-fold{fold}-best.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )
    snap = Snapshot(f"{config.output_dir}/{config.comment}-fold{fold}", config.snapshot_epochs)
    # stochastic weight averaging
    swa = SWA(
        f"{config.output_dir}/{config.comment}-fold{fold}",
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
        callbacks.append(logger)
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
        validation_steps=valid_steps,
    )

    if config.save_output:  # reload the saved best weights checkpoint
        saved_based_model = f"{config.output_dir}/{config.comment}-fold{fold}-best.h5"
        if os.path.exists(saved_based_model):
            model.load_weights(saved_based_model)
        else:
            print(f"Warning: could not find {saved_based_model}")
    if fold != "all":
        cv = model.evaluate(valid_ds, verbose=config.verbose, steps=valid_steps)
    else:
        cv = None

    return model, cv, history


def train_folds(train_filenames, folds, config=CFG, strategy=STRATEGY, summary=True):
    for fold in folds:
        if fold != "all":
            all_files = train_filenames
            train_files = [x for x in all_files if f"fold_{fold}" not in x]
            valid_files = [x for x in all_files if f"fold_{fold}" in x]
        else:
            train_files = train_filenames
            valid_files = None

        train_fold(config, fold, train_files, valid_files, strategy=strategy, summary=summary)
    return


def train():
    tf.keras.backend.clear_session()
    records_path = "/data/output/records/"
    train_filenames = glob.glob(records_path + "/*.tfrecord")

    # ds = get_tfrec_dataset(train_filenames, max_len=CFG.max_len, augment=True, batch_size=1024)
    # explore(ds)
    train_folds(train_filenames, folds=[0])
