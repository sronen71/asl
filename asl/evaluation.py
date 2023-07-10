import glob
from .config import CFG
from .training import get_dataset
from .utils import metric
from .constants import Constants
from .model import get_model
import tensorflow as tf


def eval_folds(eval_filenames, fold=0, config=CFG):
    ds = get_dataset(eval_filenames, max_len=CFG.max_len, batch_size=128)
    model = get_model(
        max_len=config.max_len,
        output_dim=config.output_dim,
        dim=config.dim,
        input_pad=Constants.INPUT_PAD,
    )
    saved_based_model = f"{config.log_path}/{config.comment}-fold{fold}-best.h5"
    model.load_weights(saved_based_model)
    lev = metric(ds, model)
    print(lev)


def eval(config=CFG):
    tf.keras.backend.clear_session()
    format = "tfrecord"
    if format == "parquet":
        data_filenames1 = sorted(glob.glob(config.input_path + "train_landmarks/*.parquet"))
        data_filenames2 = sorted(glob.glob(config.input_path + "supplemental_landmarks/*.parquet"))
    else:
        data_filenames1 = sorted(glob.glob(config.output_path + "records/*.tfrecord"))
        data_filenames2 = sorted(glob.glob(config.output_path + "sup_records/*.tfrecord"))
    data_filenames = data_filenames1  # + data_filenames2
    n = len(data_filenames)
    num_val = int(config.eval_ratio * n)
    eval_filenames = data_filenames[:num_val]

    # eval_filenames = [x for x in filenames if "fold_0" in x]
    # print(eval_filenames)

    eval_folds(eval_filenames, fold=0)
