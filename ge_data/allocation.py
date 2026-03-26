import argparse
import os
from concurrent.futures import ThreadPoolExecutor

from loguru import logger


def split_range(start, end, n, over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n  # Get the remainder of the division
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append(
                (previous, previous + current_interval - 1)
            )  # '-1' because the end is inclusive
        previous += current_interval

    return intervals


def main():
    parser = argparse.ArgumentParser()

    # entrance
    parser.add_argument('--script', type=str, required=True)
    # path
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    # data
    parser.add_argument('--dataset_name', type=str, default='ShareGPT')
    parser.add_argument('--start_row', type=int, default=0)
    parser.add_argument('--end_row', type=int, required=True)
    parser.add_argument('--num_instances', type=int, default=4)
    parser.add_argument('--gpus_per_instance', type=int, default=2)

    args = parser.parse_args()

    dataset_name = args.dataset_name
    start_row = args.start_row
    end_row = args.end_row
    num_instances = args.num_instances
    gpus_per_instance = args.gpus_per_instance

    gpu_groups = [
        ",".join(str(g) for g in range(i * gpus_per_instance, (i + 1) * gpus_per_instance))
        for i in range(num_instances)
    ]

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    intervals = split_range(start_row, end_row - 1, num_instances, over=True)

    commands = []
    for index in range(num_instances):
        start, end = intervals[index]

        cuda = gpu_groups[index]

        command = (
            f'CUDA_VISIBLE_DEVICES={cuda} python {args.script} '
            f'--start {start} '
            f'--end {end} '
            f'--index {index} '
            f'--outdir {outdir} '
            f'--data_path {args.data_path} '
            f'--model_path {args.model_path} '
            f'--dataset_name {dataset_name}'
        )
        commands.append(command)
    logger.info('\n'.join(commands))

    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        for command in commands:
            executor.submit(lambda cmd: os.system(cmd), command)


if __name__ == '__main__':
    main()
