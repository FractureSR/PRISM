import argparse
import os
import re
import shlex
from concurrent.futures import ThreadPoolExecutor

from loguru import logger


CHUNK_FILE_PATTERN = re.compile(r"^chunk_(\d+)_(\d+)\.ckpt$")


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


def count_generated_chunks(directory):
    if not directory or not os.path.isdir(directory):
        return 0

    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and CHUNK_FILE_PATTERN.match(entry.name):
                count += 1
    return count


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
        worker_hdfs_outdir = os.path.join(hdfs_outdir, str(index))
        generated_files = count_generated_chunks(worker_hdfs_outdir)
        resume_start = min(end, start + generated_files * args.samples_per_file)

        if resume_start >= end:
            logger.info(
                f"worker={index} already finished: "
                f"range=[{start}, {end}), generated_files={generated_files}"
            )
            continue

        first_gpu = index * 2
        cuda = f"{first_gpu},{first_gpu + 1}"

        logger.info(
            f"worker={index} resume from {resume_start} "
            f"(range=[{start}, {end}), generated_files={generated_files})"
        )
        command = (
            f'CUDA_VISIBLE_DEVICES={cuda} python {shlex.quote(args.script)} '
            f'--start {resume_start} '
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

    if not commands:
        logger.info('No pending work. All workers are already complete.')
        return

    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        for command in commands:
            executor.submit(lambda cmd: os.system(cmd), command)


if __name__ == '__main__':
    main()
