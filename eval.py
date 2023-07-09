import glob
from config import CFG
from train import get_tfrec_dataset
from utils import metric
from model import get_model
from train import INPUT_PAD
import tensorflow as tf


def eval_folds(eval_filenames, fold=0):
    config = CFG
    ds = get_tfrec_dataset(eval_filenames, max_len=CFG.max_len, batch_size=512)
    model = get_model(
        max_len=CFG.max_len, output_dim=CFG.output_dim, dim=CFG.dim, input_pad=INPUT_PAD
    )
    saved_based_model = f"{config.output_dir}/{config.comment}-fold{fold}-best.h5"
    model.load_weights(saved_based_model)
    lev = metric(ds, model)
    print(lev)


def main():
    tf.keras.backend.clear_session()
    records_path = "/data/output/records/"
    filenames = glob.glob(records_path + "/*.tfrecord")
    eval_filenames = [x for x in filenames if "fold_0" in x]
    # print(eval_filenames)

    eval_folds(eval_filenames, fold=0)


if __name__ == "__main__":
    main()
