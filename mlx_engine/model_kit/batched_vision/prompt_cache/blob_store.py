import io
import os
import tempfile
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_vlm.models.cache as mlx_vlm_cache
from mlx.utils import tree_unflatten


@dataclass
class _BlobExtent:
    offset: int
    length: int


class _BlobReader:
    """Read-only binary stream for one immutable blob inside the store."""

    # Minimal file-like API for MLX; the blob store owns fd lifetime.
    closed = False

    def __init__(self, fd: int, offset: int, length: int):
        self._fd = fd
        self._offset = offset
        self._length = length
        self._pos = 0

    def tell(self) -> int:
        return self._pos

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            # Absolute position within this blob.
            pos = offset
        elif whence == os.SEEK_CUR:
            # Relative to this reader's logical cursor.
            pos = self._pos + offset
        elif whence == os.SEEK_END:
            # Relative to this blob's end, not the backing file's end.
            pos = self._length + offset
        else:
            raise ValueError(f"unsupported logical blob seek mode: {whence}")

        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = min(pos, self._length)
        return self._pos

    def readinto(self, buffer) -> int:
        view = memoryview(buffer)
        read_len = min(len(view), self._length - self._pos)
        # A committed blob is immutable and must be fully readable.
        chunk = os.pread(self._fd, read_len, self._offset + self._pos)
        if len(chunk) != read_len:
            raise OSError("short read from safetensor blob store")
        view[:read_len] = chunk
        self._pos += read_len
        return read_len


class TemporarySafetensorBlobStore:
    """Single-fd temporary-file store for immutable safetensors blobs.

    The backing file is created with `tempfile.TemporaryFile`, so normal process
    shutdown closes the fd and releases storage without path cleanup.

    Records are exact-size extents. `_free_extents` is always sorted by offset
    and coalesced; free space at the file tail truncates the backing file.

    Not thread-safe: production access is owned by the prompt-cache I/O thread.
    """

    def __init__(self, directory: Path):
        self._file = tempfile.TemporaryFile(mode="w+b", dir=directory)
        self._fd = self._file.fileno()
        self._end = 0
        self._free_extents: list[_BlobExtent] = []
        self._records: dict[str, _BlobExtent] = {}

    def put(
        self,
        key: str,
        arrays: dict[str, Any],
        safetensor_metadata: dict[str, str],
    ) -> int:
        """Store one safetensors record and return its serialized byte length."""
        existing_ref = self._records.get(key)
        if existing_ref is not None:
            return existing_ref.length

        # Serialize first so the store can reserve the exact blob size.
        buffer = io.BytesIO()
        mx.save_safetensors(buffer, arrays, safetensor_metadata)
        blob = buffer.getbuffer()

        blob_len = len(blob)
        offset = self._reserve(blob_len)
        try:
            self._write_blob(offset, blob)
        except Exception:
            self._release(offset, blob_len)
            raise

        self._records[key] = _BlobExtent(
            offset=offset,
            length=blob_len,
        )
        return blob_len

    def load_record(self, key: str) -> list[Any]:
        """Load a committed record; missing keys are caller invariant errors."""
        ref = self._records[key]
        reader = _BlobReader(self._fd, ref.offset, ref.length)
        return _load_record_from_file(reader)

    def exists(self, key: str) -> bool:
        return key in self._records

    def size(self, key: str) -> int:
        """Return a committed record size; missing keys are invariant errors."""
        return self._records[key].length

    def delete(self, key: str) -> None:
        ref = self._records.pop(key, None)
        if ref is None:
            return
        self._release(ref.offset, ref.length)

    def close(self) -> None:
        self._records.clear()
        self._free_extents.clear()
        self._end = 0
        self._file.close()

    def _reserve(self, capacity: int) -> int:
        """Reserve exactly `capacity` bytes and return the blob-store offset."""
        # Try to reuse space from previously deleted records.
        for idx, extent in enumerate(self._free_extents):
            if extent.length < capacity:
                continue

            offset = extent.offset
            if extent.length == capacity:
                self._free_extents.pop(idx)
            else:
                # Keep any unused tail available for later records.
                extent.offset += capacity
                extent.length -= capacity
            return offset

        # Otherwise append a new extent to the temporary backing file.
        offset = self._end
        self._end += capacity
        return offset

    def _write_blob(self, offset: int, blob) -> None:
        """Write a complete serialized blob at a reserved blob-store offset."""
        written = 0
        view = memoryview(blob)
        while written < len(view):
            n = os.pwrite(self._fd, view[written:], offset + written)
            if n <= 0:
                raise OSError("short write to safetensor blob store")
            written += n

    def _release(self, offset: int, length: int) -> None:
        """Return an extent to the free list and shrink trailing free space."""
        if length <= 0:
            raise ValueError("released blob-store extent must have positive length")

        # Maintain the sorted/coalesced `_free_extents` invariant.
        idx = bisect_left(
            self._free_extents,
            offset,
            key=lambda extent: extent.offset,
        )
        self._free_extents.insert(idx, _BlobExtent(offset=offset, length=length))

        # Merge with the previous extent if the freed range touches it.
        if idx > 0:
            previous = self._free_extents[idx - 1]
            current = self._free_extents[idx]
            if previous.offset + previous.length == current.offset:
                previous.length += current.length
                self._free_extents.pop(idx)
                idx -= 1

        # Merge with the next extent after any previous-merge adjustment.
        if idx + 1 < len(self._free_extents):
            current = self._free_extents[idx]
            next_extent = self._free_extents[idx + 1]
            if current.offset + current.length == next_extent.offset:
                current.length += next_extent.length
                self._free_extents.pop(idx + 1)

        # If the free range reaches EOF, shrink the temporary backing file.
        while (
            self._free_extents
            and self._free_extents[-1].offset + self._free_extents[-1].length
            == self._end
        ):
            extent = self._free_extents.pop()
            self._end = extent.offset
            os.ftruncate(self._fd, self._end)


def _load_record_from_file(file_obj) -> list[Any]:
    """Deserialize one safetensors blob back into mlx-vlm cache objects."""
    arrays, safetensor_metadata = mx.load(
        file_obj,
        format="safetensors",
        return_metadata=True,
    )
    arrays = tree_unflatten(list(arrays.items()))
    cache_meta_states, cache_class_names = tree_unflatten(
        list(safetensor_metadata.items())
    )
    return [
        getattr(mlx_vlm_cache, cache_class_name).from_state(state, meta_state)
        for cache_class_name, state, meta_state in zip(
            cache_class_names, arrays, cache_meta_states
        )
    ]
