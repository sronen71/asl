from typing import List


class CFG:
    n_splits = 5
    save_output = True
    log_path = "output"
    input_path = "/kaggle/input/"
    output_path = "/kaggle/working/"

    seed = 42
    verbose = 1  # 0) silent 1) progress bar 2) one line per epoch

    # max_len = 300  # max number of frames
    max_len = 256
    replicas = 1
    lr = 5e-4 * replicas  # 5e-4
    weight_decay = 0.1
    lr_min = 1e-6
    epoch = 50  # 400
    warmup = 0.1  # 0.1
    batch_size = 64 * replicas  # 64*
    snapshot_epochs: List[int] = []
    swa_epochs: List[int] = []  # list(range(epoch//2,epoch+1))

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
