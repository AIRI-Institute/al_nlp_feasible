import os
from pathlib import Path
from hydra import initialize, compose
import json
from shutil import rmtree
from time import time


os.environ["HYDRA_CONFIG_PATH"] = ""
os.environ["HYDRA_CONFIG_NAME"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""


from active_learning.run_active_learning import run_active_learning


def test_ner():

    start_time = time()

    init_dir = os.getcwd()
    # To be able to launch from both `tests` dir and root dir
    data_path = "data"
    if "data" not in os.listdir():
        data_path = "../data"

    with initialize(config_path="../configs"):
        config = compose(
            config_name="al_ner",
            overrides=[
                f"data.path={data_path}",
                "data.dataset_name=ner_test",
                "al.num_queries=1",
                "seed=23419",
                "al.step_p_or_n=0.01",
                "al.init_p_or_n=0.01",
                "acquisition_model.training.trainer_args.fp16.training=False",
                "acquisition_model.training.batch_size_args.min_num_gradient_steps=10",
                "acquisition_model.training.trainer_args.num_epochs=5",
                "acquisition_model.training.trainer_args.serialization_dir=output",
                "hydra.run.dir=.",
                "output_dir=workdir_ner",
            ],
        )

    work_dir = config.output_dir
    Path(work_dir).mkdir(exist_ok=True)

    run_active_learning(config)

    with open(Path(work_dir) / "acquisition_metrics.json") as f:
        losses = json.load(f)["test_loss"]

    assert [round(x, 5) for x in losses] == [1.0076, 0.98358]

    os.chdir(init_dir)
    rmtree(work_dir, ignore_errors=True)

    print(f"Test passed within {time() - start_time:.1f} seconds")


def test_cls():

    start_time = time()

    init_dir = os.getcwd()
    # To be able to launch from both `tests` dir and root dir
    data_path = "data"
    if "data" not in os.listdir():
        data_path = "../data"

    with initialize(config_path="../configs"):
        config = compose(
            config_name="al_cls",
            overrides=[
                f"data.path={data_path}",
                "data.dataset_name=bbc_news",
                "al.num_queries=1",
                "seed=23419",
                "al.step_p_or_n=0.002",
                "al.init_p_or_n=0.002",
                "acquisition_model.training.trainer_args.fp16.training=False",
                "acquisition_model.training.batch_size_args.min_num_gradient_steps=10",
                "acquisition_model.training.trainer_args.num_epochs=5",
                "acquisition_model.training.trainer_args.serialization_dir=output",
                "hydra.run.dir=.",
                "output_dir=workdir_cls",
            ],
        )

    work_dir = config.output_dir
    Path(work_dir).mkdir(exist_ok=True)

    run_active_learning(config)

    with open(Path(work_dir) / "acquisition_metrics.json") as f:
        losses = json.load(f)["test_loss"]

    assert [round(x, 5) for x in losses] == [1.57866, 1.57172]

    os.chdir(init_dir)
    rmtree(work_dir, ignore_errors=True)

    print(f"Test passed within {time() - start_time:.1f} seconds")
