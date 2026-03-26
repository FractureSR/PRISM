import argparse
import os
import shlex
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
    parser.add_argument('--script', type=str)
    # path
    parser.add_argument('--local_outdir', type=str, required=True)
    parser.add_argument('--hdfs_outdir', type=str, default=None)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    # data
    parser.add_argument('--dataset_name', type=str, default='ShareGPT')
    parser.add_argument('--num_rows', type=int, default=68000)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--samples_per_file', type=int, default=20)
    parser.add_argument('--move_every_files', type=int, default=100)

    args = parser.parse_args()

    dataset_name = args.dataset_name
    num_rows = args.num_rows
    num_gpus = args.num_gpus

    if num_gpus % 2 != 0:
        raise ValueError('num_gpus must be even so each worker can use two GPUs')
    num_workers = num_gpus // 2

    local_root = args.local_outdir
    hdfs_root = args.hdfs_outdir or local_root
    local_outdir = f'{local_root}/{dataset_name}_{num_rows}'
    hdfs_outdir = f'{hdfs_root}/{dataset_name}_{num_rows}'
    os.makedirs(local_outdir, exist_ok=True)
    os.makedirs(hdfs_outdir, exist_ok=True)

    intervals = split_range(0, num_rows - 1, num_workers, over=True)

    commands = []
    for index in range(num_workers):
        start, end = intervals[index]
        first_gpu = index * 2
        cuda = f"{first_gpu},{first_gpu + 1}"

        command = (
            f'CUDA_VISIBLE_DEVICES={cuda} python {shlex.quote(args.script)} '
            f'--start {start} '
            f'--end {end} '
            f'--index {index} '
            f'--local_outdir {shlex.quote(local_outdir)} '
            f'--hdfs_outdir {shlex.quote(hdfs_outdir)} '
            f'--data_path {shlex.quote(args.data_path)} '
            f'--model_path {shlex.quote(args.model_path)} '
            f'--dataset_name {shlex.quote(dataset_name)} '
            f'--samples_per_file {args.samples_per_file} '
            f'--move_every_files {args.move_every_files}'
        )
        commands.append(command)
    logger.info('\n'.join(commands))

    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        for command in commands:
            executor.submit(lambda cmd: os.system(cmd), command)


if __name__ == '__main__':
    main()
