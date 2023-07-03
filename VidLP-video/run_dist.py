import os
import copy
import pytorch_lightning as pl
import os
import yaml

os.environ["NCCL_DEBUG"] = "INFO"
import torch
from tvd.config import ex
from tvd.modules import METERTransformerSS
from tvd.datamodules.multitask_datamodule import MTDataModule

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


@ex.main
def main(_config):
    local_rank = os.environ["LOCAL_RANK"]
    config = copy.deepcopy(_config)

    print("=" * 30)
    print("config:", config)
    print("=" * 30)

    pl.seed_everything(config["seed"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    config["world_size"] = world_size
    config["rank"] = rank
    config["nnodes"] = int(os.environ.get("NNODES", 1))
    config["num_nodes"] = config["nnodes"]

    dm = MTDataModule(config, dist=True)

    model = METERTransformerSS(config)
    exp_name = f'{config["exp_name"]}'

    os.makedirs(config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=None,  # use logger's path
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        filename="epoch_{epoch:0>3d}-step_{step:0>6d}-val_score_{val/the_metric:.2f}",
        auto_insert_metric_name=False,
    )
    if config["resume_from"] and config["fix_exp_version"]:
        version = int(config["resume_from"].split("/")[-3].split("_")[-1])
    else:
        version = None
    logger = pl.loggers.TensorBoardLogger(
        config["log_dir"],
        name=f'{exp_name}_seed{config["seed"]}_from_{config["load_path"].split("/")[-1][:-5]}',
        version=version,
    )

    config["exp_path"] = logger.root_dir

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = config["num_gpus"] if isinstance(config["num_gpus"], int) else len(config["num_gpus"])
    available_batch_size = config["per_gpu_batchsize"] * num_gpus * config["num_nodes"]
    grad_steps = max(config["batch_size"] // (available_batch_size), 1)
    print(f' Node Num: {num_gpus}, Total GPU Numbers: {num_gpus * config["num_nodes"]}')
    print(
        f' Total Batch Size: {config["batch_size"]}, Available Batch Size: {available_batch_size}, Per GPU Batch Size: {config["per_gpu_batchsize"]}, Grad Steps: {grad_steps}'
    )
    grad_steps = max(config["batch_size"] // (config["per_gpu_batchsize"] * num_gpus * config["num_nodes"]), 1)
    max_steps = config["max_steps"] if config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=config["num_gpus"],
        num_nodes=config["num_nodes"],
        precision=config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=config["prepare_data_per_node"],
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=100,
        flush_logs_every_n_steps=100,
        resume_from_checkpoint=config["resume_from"],
        weights_summary="top",
        fast_dev_run=config["fast_dev_run"],
        val_check_interval=config["val_check_interval"],
        progress_bar_refresh_rate=1 if config["debug"] else 200,
        num_sanity_val_steps=2,
    )

    if not config["test_only"]:
        trainer.fit(
            model,
            datamodule=dm,
        )
    else:
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument("--cfg", type=str, default="meter/config_tmp.json")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    config = json.load(open(args.cfg, "r"))
    main(config)
