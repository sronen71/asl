import glob
from .config import CFG
from .training import get_dataset
from .utils import global_metric
from .constants import Constants
from .model import get_model
import tensorflow as tf


def eval_folds(eval_filenames, config, experiment_id):
    ds = get_dataset(eval_filenames, config.input_path, max_len=CFG.max_len, batch_size=128)
    model = get_model(
        max_len=config.max_len,
        output_dim=config.output_dim,
        dim=config.dim,
        input_pad=Constants.INPUT_PAD,
    )
    saved_based_model = f"{config.log_path}/{config.comment}-exp{experiment_id}-best.h5"
    model.load_weights(saved_based_model)
    lev = global_metric(ds, model)
    print(lev)


def eval(config=CFG, experiment_id=0):
    tf.keras.backend.clear_session()
    if config.fp16:
        if config.is_tpu:
            policy = "mixed_bfloat16"
        else:
            policy = "mixed_float16"
    else:
        policy = "float32"
    tf.keras.mixed_precision.set_global_policy(policy)
    data_filenames = sorted(glob.glob(config.input_path + "train_landmarks/*.parquet"))
    eval_filenames = data_filenames[: config.num_eval]

    eval_folds(eval_filenames, config, experiment_id)
