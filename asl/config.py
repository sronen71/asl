import tensorflow as tf


def get_strategy():
    logical_devices = tf.config.list_logical_devices()
    # Check if TPU is available
    tpu_available = any("TPU" in device.name for device in logical_devices)
    gpu_available = any("GPU" in device.name for device in logical_devices)
    strategy = None
    if tpu_available:
        tpu = "local"
        print("connecting to TPU...")
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=tpu)
        strategy = tf.distribute.TPUStrategy(tpu)

    elif gpu_available:
        ngpu = len(tf.config.list_physical_devices("GPU"))
        print("Num GPUs Available: ", ngpu)
        if ngpu > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.get_strategy()
    replicas = strategy.num_replicas_in_sync
    print(f"REPLICAS: {replicas}")

    return strategy, replicas


class CFG:
    strategy, replicas = get_strategy()
    n_splits = 5
    save_output = True
    log_path = "logs"
    input_path = "/kaggle/input/"
    output_path = "/kaggle/working/"

    seed = 42
    verbose = 1  # 0) silent 1) progress bar 2) one line per epoch

    # max number of frames
    max_len = 256
    # max_len = 128
    replicas = 1
    lr = 2e-4 * replicas  # 5e-4
    weight_decay = 5e-5  # 4e-4
    epoch = 50  # 400
    batch_size = 64 * replicas  # 64*
    snapshot_epochs = []  # type: ignore
    swa_epochs = []  # type: ignore
    # list(range(epoch//2,epoch+1))

    fp16 = True
    # fp16 = False
    fgm = False
    awp = False
    awp_lambda = 0.2
    awp_start_epoch = 15
    dropout_start_epoch = 15
    resume = 0
    decay_type = "cosine"
    # dim = 192
    dim = 384
    comment = f"model-{dim}-seed{seed}"
    output_dim = 61
    eval_ratio = 0.1
