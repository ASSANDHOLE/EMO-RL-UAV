import contextlib
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from gym_uav import NamedDict

_NotFoundShmWrapper = NamedDict({
    'close': lambda: None,
    'unlink': lambda: None,
    'buf': memoryview(np.zeros(10, dtype=np.int32).tobytes())
})


def shm_for_name(name):
    try:
        shm = SharedMemory(name=name)
        return shm
    except FileNotFoundError:
        return _NotFoundShmWrapper


class GpuResourceScheduler:
    def __init__(self, available_devices, lock, limit_per_device=None):
        self.available_devices = available_devices
        self.gpu_count = len(available_devices)
        arr = np.zeros(self.gpu_count, dtype=np.int32)
        self.used_gpu = SharedMemory(create=True, size=arr.nbytes)
        self.buffer_name = self.used_gpu.name
        arr_b = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.used_gpu.buf)
        arr_b[:] = arr[:]
        self.limit_per_device = int(limit_per_device) if limit_per_device is not None else 999999
        self.gpu_lock = lock

    def delete(self):
        try:
            self.used_gpu.close()
            shm = shm_for_name(self.buffer_name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

    def get_gpu_id(self):
        with self.gpu_lock:
            shm = shm_for_name(self.buffer_name)
            used_gpu = np.ndarray((self.gpu_count,), dtype=np.int32, buffer=shm.buf)
            min_arg = np.argmin(used_gpu)
            if used_gpu[min_arg] >= self.limit_per_device:
                shm.close()
                return None
            used_gpu[min_arg] += 1
            shm.close()
            return self.available_devices[min_arg]

    def return_gpu_id(self, gpu_id):
        with self.gpu_lock:
            if gpu_id not in self.available_devices:
                return
            shm = shm_for_name(self.buffer_name)
            used_gpu = np.ndarray((self.gpu_count,), dtype=np.int32, buffer=shm.buf)
            idx = self.available_devices.index(gpu_id)
            used_gpu[idx] -= 1
            shm.close()

    @contextlib.contextmanager
    def context_assign_id(self):
        gpu_id = self.get_gpu_id()
        while gpu_id is None:
            time.sleep(0.1)
            gpu_id = self.get_gpu_id()
        try:
            yield gpu_id
        finally:
            self.return_gpu_id(gpu_id)
