from sacred import Experiment

ex = Experiment("METER")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "irtr": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    add_new_bos_token = False
    append_eos_token = False
    prepend_bos_token = False
    get_mlm_caption_metric = False
    frame_num_json_path = (
        "/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/frame_num.json"
    )
    gt_caption_anno_path = (
        "/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/test_coco_style_spaced.json"
    )

    # For video processing
    num_frames = 4  # input video frames
    backend = "a100"  # gpu: a100/v100/others
    draw_options_text = 0
    max_image_len = -1
    video_backbone = None

    # hybrid language modeling
    debug = False
    is_causal_mask = False
    disable_cross_modal_image_layer = False
    retrieval_views = 1
    get_cl_recall_metric = False
    skip_test_step = False
    lm3_share_weights = True
    lm3_backbone = False
    temperature = 0.05
    random_lm3_mask = False
    disable_lm3_shuffle = False
    lm3_mask_prob = 0.0
    text_encoder_from_scratch = False
    cross_att_gating = False
    cl_head_bias = True
    learnable_temperature = False

    # exp setting
    exp_name = "meter"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = (
        4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    )

    prepare_data_per_node = True
    # Image setting
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 224
    patch_size = 32
    draw_false_image = 1
    image_only = False
    resolution_before = 224

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 50
    tokenizer = ".cache/bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False  # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    num_top_layer = 6
    input_image_embed_size = 768
    input_text_embed_size = 768
    vit = "ViT-B/32"
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 1500
    warmup_steps = 0.05
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 8
    num_nodes = 1
    load_path = ""
    fix_exp_version = False
    num_workers = 8
    precision = 32


@ex.named_config
def task_mlm_itm_drama():
    exp_name = "mlm_itm_drama"
    datasets = ["drama"]  # "howto100m", "webvid",
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # max batch size for all nodes
    max_epoch = 100
    max_steps = 1500
    warmup_steps = 0.05
    max_image_len = -1  # -1/200 only use 200 image tokens for pretrain?
    val_check_interval = 1.0  # val for each 0.3 epoch, if int, is sample
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    tokenizer = ".cache/bert-base-chinese"
    vocab_size = 21128
    # video_backbone='SpaceTimeTransformer'
    vit = "SpaceTimeTransformer"
    image_size = 224


@ex.named_config
def mlm_itm():
    exp_name = "mlm_itm"
    # datasets = ["gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True


@ex.named_config
def mlm():
    exp_name = "mlm"
    # datasets = ["gcc"]
    loss_names = _loss_names({"mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True


@ex.named_config
def itm():
    exp_name = "itm"
    # datasets = ["gcc"]
    loss_names = _loss_names({"itm": 1})
    batch_size = 4096
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True


@ex.named_config
def ft_irtr_drama():
    exp_name = "finetune_irtr_drama"
    datasets = ["dramaFT"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 5
    learning_rate = 5e-6
    lr_mult_head = 5
    lr_mult_cross_modal = 5
    tokenizer = ".cache/bert-base-chinese"
    vocab_size = 21128
    input_text_embed_size = 768
    vit = "ViT-B/32"
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    input_image_embed_size = 768
    vit = "SpaceTimeTransformer"
    image_size = 224


@ex.named_config
def ft_mlm_cap_drama():
    exp_name = "mlm"
    # datasets = ["gcc"]
    loss_names = _loss_names({"mlm": 1})
    max_epoch = 30
    max_steps = None
    warmup_steps = 0.1
    whole_word_masking = True
    batch_size = 256
    per_gpu_batchsize = 64
    learning_rate = 5e-5

    add_new_bos_token = True
    append_eos_token = True
    prepend_bos_token = False
    # caption_prompt=None
    get_mlm_caption_metric = True

    is_causal_mask = True
    disable_cross_modal_image_layer = True
    datasets = ["dramaFT"]

    tokenizer = ".cache/bert-base-chinese"
    vocab_size = 21128
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    input_image_embed_size = 768
    vit = "SpaceTimeTransformer"
    image_size = 224
    max_text_len = 100


@ex.named_config
def ft_mlm_cap_dramacap():
    exp_name = "mlm"
    # datasets = ["gcc"]
    loss_names = _loss_names({"mlm": 1})
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    whole_word_masking = True
    batch_size = 256
    per_gpu_batchsize = 64
    learning_rate = 1e-5

    add_new_bos_token = True
    append_eos_token = True
    prepend_bos_token = False
    # caption_prompt=None
    get_mlm_caption_metric = True

    is_causal_mask = True
    disable_cross_modal_image_layer = True
    datasets = ["dramaPT"]

    tokenizer = ".cache/bert-base-chinese"
    vocab_size = 21128
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    input_image_embed_size = 768
    vit = "SpaceTimeTransformer"
    image_size = 224
    max_text_len = 80


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end


# vision encoder
@ex.named_config
def swin32_base224():
    vit = "swin_base_patch4_window7_224_in22k"
    patch_size = 32
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024
    resolution_before = 224


@ex.named_config
def swin32_base384():
    vit = "swin_base_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024
    resolution_before = 384


@ex.named_config
def swin32_large384():
    vit = "swin_large_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1536
    resolution_before = 384


@ex.named_config
def clip32():
    vit = "ViT-B/32"
    image_size = 224
    patch_size = 32
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768


@ex.named_config
def clip16():
    vit = "ViT-B/16"
    image_size = 224
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768


# text encoder
@ex.named_config
def text_roberta():
    tokenizer = ".cache/roberta-base"
    vocab_size = 50265
    input_text_embed_size = 768


@ex.named_config
def text_roberta_large():
    tokenizer = ".cache/roberta-large"
    vocab_size = 50265
    input_text_embed_size = 1024


# random augmentation
@ex.named_config
def imagenet_randaug():
    train_transform_keys = ["imagenet_randaug"]


@ex.named_config
def clip_randaug():
    train_transform_keys = ["clip_randaug"]
