from math import ceil
import os


def get_num_epochs_and_batch_size(
    len_train_instances: int,
    num_epochs: int,
    current_batch_size: int,
    adjust_batch_size: bool,
    adjust_num_epochs: bool,
    min_num_gradient_steps: int,
    min_batch_size: int = 4,
):
    # Get num cuda-s for training parallelization
    # num_cudas = max(len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')), 1)
    # min_batch_size = ceil(min_batch_size / num_cudas) * num_cudas
    # If num_epochs < 1, num_epochs will be adjusted instead of batch_size
    if adjust_batch_size and num_epochs >= 1:
        required_steps_per_epoch = ceil(min_num_gradient_steps / num_epochs)
        current_steps_per_epoch = ceil(len_train_instances / current_batch_size)
        if current_steps_per_epoch < required_steps_per_epoch:
            fraction = required_steps_per_epoch / current_steps_per_epoch
            candidates = [
                ceil(current_batch_size / fraction),
                round(current_batch_size / fraction),
                int(current_batch_size / fraction),
                int(current_batch_size / fraction) - 1,
            ]

            while current_steps_per_epoch < required_steps_per_epoch:
                batch_size = candidates.pop(
                    0
                )  # ceil(candidates.pop(0) / num_cudas) * num_cudas

                if (batch_size < min_batch_size) or len(candidates) == 0:
                    batch_size = min_batch_size  # ceil(min_batch_size / num_cudas)
                    if adjust_num_epochs:
                        current_steps_per_epoch = ceil(len_train_instances / batch_size)
                        num_epochs = ceil(
                            required_steps_per_epoch
                            / current_steps_per_epoch
                            * num_epochs
                        )
                    break

                current_steps_per_epoch = ceil(len_train_instances / batch_size)

            return num_epochs, batch_size

    batch_size = current_batch_size  # ceil(current_batch_size / num_cudas)
    return num_epochs, batch_size


def get_train_constants(
    len_train_instances: int,
    num_epochs: int,
    current_batch_size: int,
    adjust_batch_size: bool,
    adjust_num_epochs: bool,
    min_num_gradient_steps: int,
    warmup_steps_factor: float,
    min_batch_size: int = 4,
    dev_size: float = 0.0,
):
    # If we possess a validation sample, we use early stopping and
    # train until the model starts overfitting
    if dev_size > 0:
        adjust_num_epochs = False
        num_epochs *= 2
    num_epochs, batch_size = get_num_epochs_and_batch_size(
        len_train_instances,
        num_epochs,
        current_batch_size,
        adjust_batch_size,
        adjust_num_epochs,
        min_num_gradient_steps,
        min_batch_size,
    )
    steps_per_epoch = ceil(len_train_instances / batch_size)
    scheduler_warmup_steps = int(num_epochs * steps_per_epoch * warmup_steps_factor)

    return batch_size, num_epochs, steps_per_epoch, scheduler_warmup_steps
