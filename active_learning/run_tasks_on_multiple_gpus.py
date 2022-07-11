import hydra
import os
import logging
import time
from numpy.random import randint

from active_learning.utils import utils_tasks as utils


log = logging.getLogger(__name__)


def run_task(task):
    log.info(f"Task name: {task.name}")
    task_args = task.args if "args" in task else ""
    command = f"CUDA_VISIBLE_DEVICES={','.join(utils.WORKER_CUDA_DEVICES)} HYDRA_CONFIG_PATH={task.config_path} HYDRA_CONFIG_NAME={task.config_name} {task.environ} python {task.command} {task_args} +repeat={task.repeat}"
    log.info(f"Command: {command}")
    # To prevent mixing of log in the terminal
    time.sleep(randint(1_000) / 1_000)
    ret = os.system(command)
    ret = str(ret)
    log.info(f'Task "{task.name}" finished with return code: {ret}.')
    return ret


@hydra.main(config_path=os.environ["HYDRA_CONFIG_PATH"], config_name=os.environ.get("HYDRA_EXP_CONFIG_NAME", "config"))
def main(configs):
    os.chdir(hydra.utils.get_original_cwd())
    utils.run_tasks(configs, run_task)


if __name__ == "__main__":
    main()
