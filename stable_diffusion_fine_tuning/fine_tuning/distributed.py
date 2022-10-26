import os
from stable_diffusion_fine_tuning.main import ExperimentConfig
import uuid
from typing import Callable
from functools import partial

import torch
from tf_yarn.pytorch import (
    run_on_yarn, TaskSpec, NodeLabel, PytorchExperiment,
    DataLoaderArgs
)
import torch.distributed as dist

from stable_diffusion_fine_tuning.model import load_stable_diffusion_pipeline
from stable_diffusion_fine_tuning.fine_tuning.dataset import WrapperDataset
from stable_diffusion_fine_tuning.fine_tuning.main import run_with_datasets


def run_distributed(
    model: torch.nn.Module,
    trainloader: torch.utils.data.dataloader.DataLoader,
    device: str,
    rank: int,
    tb_writer,
    config: ExperimentConfig
):
    config.device = device
    run_with_datasets(
        pipeline=model,
        train_dataloader=trainloader,
        test_dataloader=None,
        config=config
    )


# TODO: Fix duplicated params with ExperimentConfig
def get_experiment_fn(
    pipeline_hdfs_path: str,
    dataset_fn: Callable[[], torch.utils.data.Dataset],
    batch_size: int,
    experiment_config: ExperimentConfig
):
    def _experiment_fn():
        download_dir = os.path.join(".", str(uuid.uuid4()))
        pipe = load_stable_diffusion_pipeline(pipeline_hdfs_path=pipeline_hdfs_path, download_dir=download_dir)
        
        dataset, size = dataset_fn()
        if isinstance(dataset, torch.utils.data.IterableDataset):
            # TODO: add sanity check to verify that shuffle is enabled
            num_workers = dist.get_world_size() if dist.is_initialized() else 1
            dataset = WrapperDataset(dataset, size, num_workers)

        return PytorchExperiment(
            model=pipe,
            main_fn=partial(run_distributed, experiment_config=experiment_config),
            train_dataset=dataset,
            dataloader_args=DataLoaderArgs(batch_size=batch_size, num_workers=8, pin_memory=False),
            n_workers_per_executor=2
        )
    return _experiment_fn


def run_distributed(
    pipeline_hdfs_path: str,
    dataset_fn: Callable[[], torch.utils.data.Dataset],
    batch_size: int,
    experiment_config: ExperimentConfig,
    pyenv_zip_path: str,
    n_executors: int = 2,
    vcores_per_executor: int = 80,
    memory_per_executor_in_b: int = 72*2**10,
):
    run_on_yarn(
        experiment_fn=get_experiment_fn(pipeline_hdfs_path, dataset_fn, batch_size, experiment_config),
        task_specs={
            "worker": TaskSpec(
                memory=memory_per_executor_in_b, vcores=vcores_per_executor, instances=n_executors, label=NodeLabel.GPU
            )
        },
        queue="ml-gpu",
        pyenv_zip_path=pyenv_zip_path
    )