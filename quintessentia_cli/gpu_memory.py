"""Conservative startup budget from CUDA and the matching GPU's NVML counter.

On the tested Windows/WDDM host CUDA's free counter did not reflect a second
process's allocation. Use the lower valid reading; never match NVML by an ordinal
that CUDA_VISIBLE_DEVICES can remap. NVML remains optional for CPU installations.
"""
import ctypes
import os
from pathlib import Path
import threading

_LOCK = threading.Lock()


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def _load_nvml_library():
    candidates = (
        [
            str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvml.dll"),
            "nvml.dll",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
        ]
        if os.name == "nt"
        else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    )
    last_exc = None
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError as exc:
            last_exc = exc
    raise OSError(f"NVML library unavailable: {last_exc}")


def _nvml_memory(uuid):
    lib = _load_nvml_library()
    def check(code):
        if code:
            raise OSError(f"NVML returned status {code}")
    with _LOCK:
        check(lib.nvmlInit_v2())
        try:
            handle = ctypes.c_void_p()
            get_handle = lib.nvmlDeviceGetHandleByUUID
            get_handle.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
            get_handle.restype = ctypes.c_int
            check(get_handle(str(uuid).encode("ascii"), ctypes.byref(handle)))
            get_memory = lib.nvmlDeviceGetMemoryInfo
            get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Memory)]
            get_memory.restype = ctypes.c_int
            memory = _Memory()
            check(get_memory(handle, ctypes.byref(memory)))
            return int(memory.free), int(memory.total)
        finally:
            lib.nvmlShutdown()


class PhysicalVRAMMonitor:
    """Persistent, UUID-matched (~0.02 ms) physical GPU VRAM monitor.

    Single canonical NVML reader shared across startup planning and runtime
    containment (`StanokContainmentContour`). Matches the CUDA device by its
    GPU UUID (never a bare ordinal that `CUDA_VISIBLE_DEVICES` can remap) and
    falls back cleanly to `torch.cuda.mem_get_info` when NVML is absent.
    """

    def __init__(self, device=None):
        import torch

        self.device = torch.device("cuda:0" if device is None else device)
        self._lock = threading.Lock()
        self._lib = None
        self._handle = None
        self._get_memory = None
        self._get_utilization = None
        self.nvml_available = False
        self.gpu_uuid = None

        if self.device.type != "cuda" or not torch.cuda.is_available():
            return

        uuid_obj = getattr(torch.cuda.get_device_properties(self.device), "uuid", None)
        if uuid_obj is not None:
            uuid_str = str(uuid_obj)
            if not uuid_str.startswith("GPU-"):
                uuid_str = f"GPU-{uuid_str}"
            self.gpu_uuid = uuid_str
            opened = self._try_open_nvml_by_uuid(uuid_str)
            if opened is not None:
                self._lib, self._handle, self._get_memory, self._get_utilization = opened
                self.nvml_available = True

    @staticmethod
    def _try_open_nvml_by_uuid(uuid_str):
        try:
            lib = _load_nvml_library()
            if lib.nvmlInit_v2() != 0:
                return None
            handle = ctypes.c_void_p()
            get_handle = lib.nvmlDeviceGetHandleByUUID
            get_handle.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
            get_handle.restype = ctypes.c_int
            # Try exact UUID string and stripped variant (NVIDIA drivers accept "GPU-<uuid>")
            ok = False
            for candidate in (uuid_str, uuid_str.removeprefix("GPU-")):
                if get_handle(candidate.encode("ascii"), ctypes.byref(handle)) == 0:
                    ok = True
                    break
            if not ok:
                lib.nvmlShutdown()
                return None
            get_memory = lib.nvmlDeviceGetMemoryInfo
            get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Memory)]
            get_memory.restype = ctypes.c_int
            get_utilization = None
            try:
                fn = lib.nvmlDeviceGetUtilizationRates
                fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Utilization)]
                fn.restype = ctypes.c_int
                get_utilization = fn
            except AttributeError:
                get_utilization = None
            return lib, handle, get_memory, get_utilization
        except (OSError, AttributeError, UnicodeError):
            return None

    def query(self):
        """Return `(free_bytes, total_bytes, source)` where source is 'nvml' or 'cuda'."""
        import torch

        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0, 0, "cpu"
        with self._lock:
            if self.nvml_available and self._lib is not None and self._get_memory is not None:
                mem = _Memory()
                if self._get_memory(self._handle, ctypes.byref(mem)) == 0:
                    free_b, total_b = int(mem.free), int(mem.total)
                    if total_b > 0 and 0 <= free_b <= total_b:
                        return free_b, total_b, "nvml"
        fb, tb = torch.cuda.mem_get_info(self.device)
        return int(fb), int(tb), "cuda"

    def query_utilization(self):
        """Return `(gpu_util_pct, mem_util_pct, source)` in [0, 100] via NVML or `(None, None, 'unavailable')`."""
        if self.device.type != "cuda":
            return None, None, "cpu"
        with self._lock:
            if self.nvml_available and self._lib is not None and self._get_utilization is not None:
                util = _Utilization()
                if self._get_utilization(self._handle, ctypes.byref(util)) == 0:
                    g = max(0, min(100, int(util.gpu)))
                    m = max(0, min(100, int(util.memory)))
                    return float(g), float(m), "nvml"
        return None, None, "unavailable"

    def close(self):
        with self._lock:
            if self.nvml_available and self._lib is not None:
                try:
                    self._lib.nvmlShutdown()
                except Exception:
                    pass
            self._lib = None
            self._handle = None
            self._get_memory = None
            self._get_utilization = None
            self.nvml_available = False


def measure_gpu_memory(device=None):
    import torch
    free, total = (int(n) for n in torch.cuda.mem_get_info(device))
    result = dict(cuda_free_bytes=free, cuda_total_bytes=total,
                  free_bytes=free, total_bytes=total, source="cuda_only")
    try:
        uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        if uuid is None:
            raise OSError("CUDA did not expose a GPU UUID for NVML matching")
        physical_free, physical_total = _nvml_memory(uuid)
        if physical_total <= 0 or not 0 <= physical_free <= physical_total:
            raise OSError("Invalid NVML memory counters")
        result.update(physical_free_bytes=physical_free, physical_total_bytes=physical_total,
                      free_bytes=min(free, physical_free), total_bytes=min(total, physical_total),
                      source="cuda_nvml_min")
    except (OSError, AttributeError, UnicodeError) as exc:
        result["physical_counter_unavailable"] = str(exc)
    return result
