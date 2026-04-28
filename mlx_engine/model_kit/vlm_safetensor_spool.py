import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import mlx.core as mx
import mlx_lm.models.cache as mlx_lm_cache
from mlx.utils import tree_unflatten


class _SpoolCapacityExceeded(OSError):
    pass


@dataclass
class _StoredSafetensorRef:
    offset: int
    length: int


class _SpoolReader:
    def __init__(self, fd: int, offset: int, length: int):
        self._fd = fd
        self._offset = offset
        self._length = length
        self._pos = 0
        self._closed = False
        self.name = "spool.safetensors"

    @property
    def closed(self) -> bool:
        return self._closed

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            pos = offset
        elif whence == os.SEEK_CUR:
            pos = self._pos + offset
        elif whence == os.SEEK_END:
            pos = self._length + offset
        else:
            raise ValueError(f"unsupported seek mode: {whence}")

        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = min(pos, self._length)
        return self._pos

    def readinto(self, buffer) -> int:
        view = memoryview(buffer)
        remaining = self._length - self._pos
        read_len = min(len(view), remaining)
        total_read = 0
        while total_read < read_len:
            chunk = os.pread(
                self._fd,
                read_len - total_read,
                self._offset + self._pos + total_read,
            )
            if not chunk:
                break
            view[total_read : total_read + len(chunk)] = chunk
            total_read += len(chunk)

        self._pos += total_read
        return total_read


class _SpoolWriter:
    def __init__(self, fd: int, offset: int, capacity: int):
        self._fd = fd
        self._offset = offset
        self._capacity = capacity
        self._pos = 0
        self._closed = False
        self.name = "spool.safetensors"

    @property
    def closed(self) -> bool:
        return self._closed

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            pos = offset
        elif whence == os.SEEK_CUR:
            pos = self._pos + offset
        elif whence == os.SEEK_END:
            pos = self._capacity + offset
        else:
            raise ValueError(f"unsupported seek mode: {whence}")

        if not 0 <= pos <= self._capacity:
            raise _SpoolCapacityExceeded("spool write seek exceeded capacity")
        self._pos = pos
        return self._pos

    def write(self, data) -> int:
        view = memoryview(data)
        if self._pos + len(view) > self._capacity:
            raise _SpoolCapacityExceeded("spool write exceeded capacity")

        written = 0
        while written < len(view):
            n = os.pwrite(
                self._fd,
                view[written:],
                self._offset + self._pos + written,
            )
            if n <= 0:
                raise OSError("short write to safetensor spool")
            written += n

        self._pos += written
        return written


class TempSafetensorSpool:
    # One anonymous backing file keeps fd usage fixed while still storing each
    # safetensors payload as an immutable blob.
    def __init__(self, directory: Path):
        self._file = tempfile.TemporaryFile(mode="w+b", dir=directory)
        self._fd = self._file.fileno()
        self._lock = Lock()
        self._end = 0
        self._free_extents: list[tuple[int, int]] = []
        self._records: dict[str, _StoredSafetensorRef] = {}

    def put(
        self,
        key: str,
        arrays: dict[str, Any],
        metadata: dict[str, str],
    ) -> int:
        with self._lock:
            existing_ref = self._records.get(key)
            if existing_ref is not None:
                return existing_ref.length

        min_capacity = self._safetensor_size_floor(arrays)
        preferred_capacity = self._estimate_safetensor_size(arrays, metadata)
        while True:
            offset, reserved_len = self._reserve(
                min_capacity,
                preferred_capacity,
            )
            writer = _SpoolWriter(self._fd, offset, reserved_len)
            try:
                mx.save_safetensors(writer, arrays, metadata)
            except _SpoolCapacityExceeded:
                self._release_reserved(offset, reserved_len)
                min_capacity = reserved_len + 1
                preferred_capacity = max(preferred_capacity * 2, min_capacity)
                continue
            except Exception:
                self._release_reserved(offset, reserved_len)
                raise

            actual_len = writer.tell()
            return self._commit_record(
                key,
                offset,
                reserved_len,
                actual_len,
            )

    def load_prompt_cache(self, key: str) -> list[Any]:
        with self._lock:
            ref = self._records[key]
            reader = _SpoolReader(self._fd, ref.offset, ref.length)

        return _load_prompt_cache_from_file(reader)

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._records

    def size(self, key: str) -> int:
        with self._lock:
            ref = self._records.get(key)
            return ref.length if ref is not None else 0

    def record_sizes(self) -> list[int]:
        with self._lock:
            return sorted(ref.length for ref in self._records.values())

    def delete(self, key: str) -> None:
        with self._lock:
            ref = self._records.pop(key, None)
            if ref is None:
                return
            self._release_locked(ref.offset, ref.length)

    def close(self) -> None:
        with self._lock:
            self._records.clear()
            self._free_extents.clear()
            self._end = 0
            file_obj = self._file

        file_obj.close()

    def _reserve(
        self,
        min_capacity: int,
        preferred_capacity: int,
    ) -> tuple[int, int]:
        with self._lock:
            for idx, (offset, free_len) in enumerate(self._free_extents):
                if free_len < min_capacity:
                    continue

                self._free_extents.pop(idx)
                return offset, free_len

            reserved_len = max(min_capacity, preferred_capacity)
            offset = self._end
            self._end += reserved_len
            return offset, reserved_len

    def _release_reserved(self, offset: int, reserved_len: int) -> None:
        with self._lock:
            self._release_locked(offset, reserved_len)

    def _release_locked(self, offset: int, length: int) -> None:
        if length <= 0:
            return

        self._free_extents.append((offset, length))
        self._free_extents.sort()

        merged: list[tuple[int, int]] = []
        for free_offset, free_len in self._free_extents:
            if merged and merged[-1][0] + merged[-1][1] == free_offset:
                previous_offset, previous_len = merged[-1]
                merged[-1] = (previous_offset, previous_len + free_len)
            else:
                merged.append((free_offset, free_len))
        self._free_extents = merged

        while (
            self._free_extents
            and self._free_extents[-1][0] + self._free_extents[-1][1] == self._end
        ):
            offset, length = self._free_extents.pop()
            self._end = offset
            os.ftruncate(self._fd, self._end)

    def _commit_record(
        self,
        key: str,
        offset: int,
        reserved_len: int,
        actual_len: int,
    ) -> int:
        with self._lock:
            old_ref = self._records.get(key)
            if old_ref is not None:
                self._release_locked(offset, reserved_len)
                return old_ref.length

            self._records[key] = _StoredSafetensorRef(
                offset=offset,
                length=actual_len,
            )
            self._release_locked(offset + actual_len, reserved_len - actual_len)
            return actual_len

    def _estimate_safetensor_size(
        self,
        arrays: dict[str, Any],
        metadata: dict[str, str],
    ) -> int:
        array_bytes = sum(_array_nbytes(array) for array in arrays.values())
        metadata_bytes = sum(len(key) + len(value) for key, value in metadata.items())
        header_padding = max(64 * 1024, array_bytes // 64)
        return array_bytes + metadata_bytes + header_padding

    def _safetensor_size_floor(self, arrays: dict[str, Any]) -> int:
        return sum(_array_nbytes(array) for array in arrays.values()) + 8


def _array_nbytes(array: Any) -> int:
    nbytes = getattr(array, "nbytes", 0)
    if callable(nbytes):
        return int(nbytes())
    return int(nbytes)


def _load_prompt_cache_from_file(file_obj):
    arrays, cache_metadata = mx.load(file_obj, return_metadata=True)
    arrays = tree_unflatten(list(arrays.items()))
    cache_metadata = tree_unflatten(list(cache_metadata.items()))
    info, _metadata, classes = cache_metadata
    return [
        getattr(mlx_lm_cache, cache_class).from_state(state, meta_state)
        for cache_class, state, meta_state in zip(classes, arrays, info)
    ]
