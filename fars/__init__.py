class LipSimConfig:
    mode = 'fars'
    train_dir = 'fars'
    data_dir = '.'
    dataset = 'night'
    shift_data = True
    normalize_data = True  # check
    epochs = 10
    logging_verbosity = 'INFO'
    model_name = 'small'
    batch_size = 32
    depth = 0
    num_channels = 0
    depth_linear = 0
    n_features = 0
    conv_size = 5
    first_layer = 'padding_channels'
    last_layer = 'pooling_linear'
    teacher_model_name = 'ensemble'  # 'open_clip_vitb32'
    local = True
    constraint = ''
    partition = 'gpu_p5'
    ngpus = 1
    nnodes = 1
    qos = ''
    account = ''
    timeout = 1
    requires_bias = False