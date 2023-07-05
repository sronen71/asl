from typing import List


class CFG:
    n_splits = 5
    save_output = True
    # output_dir = "/kaggle/working"
    output_dir = "output"

    seed = 42
    verbose = 2  # 0) silent 1) progress bar 2) one line per epoch

    max_len = 400
    replicas = 1
    lr = 5e-4 * replicas
    weight_decay = 0.1
    lr_min = 1e-6
    epoch = 1  # 400
    warmup = 0  # 0.1
    batch_size = 8 * replicas  # 64*
    snapshot_epochs: List[int] = []
    swa_epochs: List[int] = []  # list(range(epoch//2,epoch+1))

    # fp16 = True
    fp16 = False
    fgm = False
    awp = False
    awp_lambda = 0.2
    awp_start_epoch = 15
    dropout_start_epoch = 15
    resume = 0
    decay_type = "cosine"
    dim = 192
    comment = f"model-{dim}-seed{seed}"
    output_dim = 61
