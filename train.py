import os
import numpy as np
import glob
from preprocess import ROWS_PER_FRAME, get_char_dict
from utils import POINT_LANDMARKS, LNOSE, RNOSE, LHAND, RHAND, LLIP, RLIP, LPOSE, RPOSE, LEYE, REYE
from visualize import visualize_train

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402

MAX_STRING_LEN = 50
MAX_LEN = 400
COORDINATES_PAD = -100.0
char_dict = get_char_dict()
LABEL_PAD = char_dict["PAD"]

NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6 * NUM_NODES


def interp1d_(x, target_len, method="random"):
    target_len = tf.maximum(1, target_len)
    if method == "random":
        if tf.random.uniform(()) < 0.33:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]), "bilinear")
        elif tf.random.uniform(()) < 0.5:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]), "bicubic")
        else:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]), "nearest")
    else:
        x = tf.image.resize(x, (target_len, tf.shape(x)[1]), method)
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
    def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs):
        if tf.rank(inputs) == 3:
            x = inputs[None, ...]
        else:
            x = inputs

        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2)  # N,T,P,C
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
                tf.reshape(x, (-1, length, 2 * len(self.point_landmarks))),
                tf.reshape(dx, (-1, length, 2 * len(self.point_landmarks))),
                tf.reshape(dx2, (-1, length, 2 * len(self.point_landmarks))),
            ],
            axis=-1,
        )
        # x = tf.reshape(x, (-1, length, 2 * len(self.point_landmarks)))
        x = tf.where(tf.math.is_nan(x), tf.constant(0.0, x.dtype), x)

        return x


def flip_lr(x):
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
    shear=(-0.15, 0.15),
    shift=(-0.1, 0.1),
    degree=(-30, 30),
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


def temporal_crop(x, max_length=MAX_LEN):
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
    if tf.random.uniform(()) < 0.8 or always:
        x = resample(x, (0.5, 1.5))
    if tf.random.uniform(()) < 0.5 or always:
        x = flip_lr(x)
    if max_len is not None:
        x = temporal_crop(x, max_len)
    if tf.random.uniform(()) < 0.75 or always:
        x = spatial_random_affine(x)
    if tf.random.uniform(()) < 0.2 or always:
        x = temporal_mask(x)
    if tf.random.uniform(()) < 0.3 or always:
        x = spatial_mask(x)
    return x


def filter_nans(frames):
    return frames[~np.isnan(frames).all(axis=(-2, -1))]


def filter_nans_tf(x, ref_point=POINT_LANDMARKS):
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
        tf.sparse.to_dense(features["coordinates"]), (-1, ROWS_PER_FRAME, 3)
    )
    out["label"] = tf.sparse.to_dense(features["label"])
    out["sequence_id"] = features["sequence_id"][0]
    return out


def preprocess(x, augment=False, max_len=MAX_LEN):
    coord = x["coordinates"]
    coord = filter_nans_tf(coord)
    if augment:
        coord = augment_fn(coord, max_len=max_len)
    coord = tf.ensure_shape(coord, (None, ROWS_PER_FRAME, 3))
    return tf.cast(Preprocess(max_len=max_len)(coord)[0], tf.float32), x["label"], x["sequence_id"]


def get_tfrec_dataset(
    tfrecords,
    batch_size=64,
    max_len=MAX_LEN,
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
                tf.constant(COORDINATES_PAD, dtype=tf.float32),
                tf.constant(LABEL_PAD, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64),
            ),
            padded_shapes=([max_len, CHANNELS], [MAX_STRING_LEN], ()),
            drop_remainder=drop_remainder,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def main():
    records_path = "output/records/"
    train_file_names = glob.glob(records_path + "/*.tfrecord")
    print(train_file_names)
    augment = True
    # augment = False
    ds = get_tfrec_dataset(train_file_names, augment=augment, batch_size=1024)
    for feature, label, sequence_id in ds:
        feature = feature.numpy()
        label = label.numpy()
        sequence_id = sequence_id.numpy()
        # print(sequence_id, feature.shape, label.shape)
        for i in range(3):
            N = feature.shape[-1]
            coordinates = feature[i, :].reshape(-1, N // 6, 6)
            coordinates = coordinates[:, :, :2]
            visualize_train(sequence_id[i], coordinates, label[i, :])


if __name__ == "__main__":
    main()
