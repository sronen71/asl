import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_strategy():
    logical_devices = tf.config.list_logical_devices()
    # Check if TPU is available
    tpu_available = any("TPU" in device.name for device in logical_devices)
    gpu_available = any("GPU" in device.name for device in logical_devices)
    strategy = None
    is_tpu = False
    if tpu_available:
        tpu = "local"
        print("connecting to TPU...")
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        is_tpu = True

    elif gpu_available:
        gpus = tf.config.list_physical_devices("GPU")
        # for gpu in gpus:
        #    tf.config.experimental.set_memory_growth(gpu, True)
        ngpu = len(gpus)
        print("Num GPUs Available: ", ngpu)
        if ngpu > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()

    else:
        strategy = tf.distribute.get_strategy()
    replicas = strategy.num_replicas_in_sync
    print(f"REPLICAS: {replicas}")

    return strategy, replicas, is_tpu


class CFG:
    strategy, replicas, is_tpu = get_strategy()
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
    lr = 5e-4 * replicas  # 5e-4
    weight_decay = 1e-4  # 4e-4
    epochs = 80  # 400
    batch_size = 128 * replicas
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
    dim = 384
    comment = f"model-{dim}-seed{seed}"
    output_dim = 61
    num_eval = 6
