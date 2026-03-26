import os
import re
import shutil
import tempfile

import torch


CHUNK_FILE_PATTERN = re.compile(r"^chunk_(\d+)_(\d+)\.ckpt$")


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def atomic_torch_save(data, path):
    ensure_dir(os.path.dirname(path))
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path),
        prefix=".tmp_chunk_",
        suffix=".ckpt",
    )
    os.close(fd)
    try:
        torch.save(data, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _iter_chunk_indices(directory):
    if not directory or not os.path.isdir(directory):
        return

    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            match = CHUNK_FILE_PATTERN.match(entry.name)
            if match:
                yield int(match.group(1))


def get_next_chunk_index(*directories):
    max_index = -1
    for directory in directories:
        for index in _iter_chunk_indices(directory):
            max_index = max(max_index, index)
    return max_index + 1


class ChunkWriter:
    def __init__(
        self,
        local_dir,
        hdfs_dir=None,
        samples_per_file=20,
        move_every_files=100,
    ):
        self.local_dir = local_dir
        self.hdfs_dir = hdfs_dir or local_dir
        self.samples_per_file = samples_per_file
        self.move_every_files = move_every_files
        self.buffer = []
        self.pending_local_files = []

        ensure_dir(self.local_dir)
        ensure_dir(self.hdfs_dir)
        self.chunk_idx = get_next_chunk_index(self.local_dir, self.hdfs_dir)

    @property
    def should_move_to_hdfs(self):
        return os.path.abspath(self.local_dir) != os.path.abspath(self.hdfs_dir)

    def add(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) >= self.samples_per_file:
            self.flush_chunk()

    def flush_chunk(self):
        if not self.buffer:
            return

        sample_count = len(self.buffer)
        filename = f"chunk_{self.chunk_idx:08d}_{sample_count:02d}.ckpt"
        local_path = os.path.join(self.local_dir, filename)
        payload = {
            "format": "prism_chunk_v1",
            "samples": self.buffer,
        }
        atomic_torch_save(payload, local_path)

        self.pending_local_files.append(local_path)
        self.buffer = []
        self.chunk_idx += 1

        if self.should_move_to_hdfs and len(self.pending_local_files) >= self.move_every_files:
            self.move_pending_files()

    def move_pending_files(self):
        if not self.should_move_to_hdfs or not self.pending_local_files:
            return

        for local_path in self.pending_local_files:
            target_path = os.path.join(self.hdfs_dir, os.path.basename(local_path))
            shutil.move(local_path, target_path)
        self.pending_local_files = []

    def close(self):
        self.flush_chunk()
        self.move_pending_files()
