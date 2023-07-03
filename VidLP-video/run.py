import os
import copy
import pytorch_lightning as pl

os.environ["NCCL_DEBUG"] = "INFO"

from tvd.config import ex
from tvd.modules import METERTransformerSS
from tvd.datamodules.multitask_datamodule import MTDataModule


@ex.automain
def main(_config):
    config = copy.deepcopy(_config)
    print("=" * 30)
    print("config:", config)
    print("=" * 30)

    config["world_size"] = int(os.environ.get("WORLD_SIZE", 1))
    config["rank"] = int(os.environ.get("RANK", 1))
    config["nnodes"] = int(os.environ.get("NNODES", 1))
    pl.seed_everything(config["seed"])
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
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=config["resume_from"],
        weights_summary="top",
        fast_dev_run=config["fast_dev_run"],
        val_check_interval=config["val_check_interval"],
        num_sanity_val_steps=2,
    )

    if not config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
