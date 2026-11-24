"""Native C-extension CUDA driver launcher for Quintessentia hot paths.

Eliminates the ~115 us/call Python+ctypes `cuLaunchKernel` overhead (which starved the GPU
between small decode kernels), dropping kernel submission to ~2-3 us/call and enabling
single-launch constant-memory batched MoE expert execution (`grid=(tiles, 1, H)`).

Auto-compiles once on Linux/WSL2 when `g++` and `Python.h` are present, cached by source hash
in `~/.cache/quintessentia/`. Falls back transparently to pure Python `ctypes` if unavailable.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig
import threading
from typing import Any, Optional

_C_SOURCE = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>

typedef int (*cuLaunchKernel_fn)(
    void* f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    void* hStream,
    void** kernelParams,
    void** extra
);

static cuLaunchKernel_fn g_cuLaunchKernel = NULL;

static int ensure_cuda_driver(void) {
    if (g_cuLaunchKernel != NULL) return 1;
    const char* libs[] = {
        "/usr/lib/wsl/lib/libcuda.so.1",
        "libcuda.so.1",
        "libcuda.so",
        NULL
    };
    for (int i = 0; libs[i] != NULL; ++i) {
        void* h = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
        if (!h) continue;
        void* sym = dlsym(h, "cuLaunchKernel");
        if (sym) {
            g_cuLaunchKernel = (cuLaunchKernel_fn)sym;
            return 1;
        }
    }
    return 0;
}

/* Kernel #6 single projection: pure_mirror_diff_q1_gemv_f32 */
static PyObject* py_launch_diff_q1_gemv(PyObject* self, PyObject* args) {
    unsigned long long fn_ptr, stream_ptr;
    unsigned int gx, gy, smem;
    unsigned long long px, pxl, pyl, ppk, pqs, pqz, prs, pcs, pbs, pyo, pup, pdf;
    int M, N, K, GS, Pm, Pn, ob, ss;
    float tol;
    if (!PyArg_ParseTuple(
            args, "KKIIIKKKKKKKKKKKKiiiiiifii",
            &fn_ptr, &stream_ptr, &gx, &gy, &smem,
            &px, &pxl, &pyl, &ppk, &pqs, &pqz, &prs, &pcs, &pbs, &pyo, &pup, &pdf,
            &M, &N, &K, &GS, &Pm, &Pn, &tol, &ob, &ss)) {
        return NULL;
    }
    if (!ensure_cuda_driver()) {
        PyErr_SetString(PyExc_RuntimeError, "libcuda cuLaunchKernel not available");
        return NULL;
    }
    void* a0 = (void*)(uintptr_t)px;
    void* a1 = (void*)(uintptr_t)pxl;
    void* a2 = (void*)(uintptr_t)pyl;
    void* a3 = (void*)(uintptr_t)ppk;
    void* a4 = (void*)(uintptr_t)pqs;
    void* a5 = (void*)(uintptr_t)pqz;
    void* a6 = (void*)(uintptr_t)prs;
    void* a7 = (void*)(uintptr_t)pcs;
    void* a8 = (void*)(uintptr_t)pbs;
    void* a9 = (void*)(uintptr_t)pyo;
    void* a10 = (void*)(uintptr_t)pup;
    void* a11 = (void*)(uintptr_t)pdf;
    void* kparams[21] = {
        &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11,
        &M, &N, &K, &GS, &Pm, &Pn, &tol, &ob, &ss
    };
    int rc = g_cuLaunchKernel(
        (void*)(uintptr_t)fn_ptr,
        gx, gy, 1, 256, 1, 1, smem,
        (void*)(uintptr_t)stream_ptr, kparams, NULL
    );
    return PyLong_FromLong((long)rc);
}

/* Kernel #6c batched MoE layer: up to 16 experts passed by value in CUDA constant memory (0 H2D!) */
#define MAX_MOE_BATCH 16
typedef struct {
    unsigned long long x_last[MAX_MOE_BATCH];
    unsigned long long y_last[MAX_MOE_BATCH];
    unsigned long long packed_lo[MAX_MOE_BATCH];
    unsigned long long qscale[MAX_MOE_BATCH];
    unsigned long long qzero[MAX_MOE_BATCH];
    unsigned long long row_scale[MAX_MOE_BATCH];
    unsigned long long col_scale[MAX_MOE_BATCH];
    int p_max[MAX_MOE_BATCH];
    int shared_scale[MAX_MOE_BATCH];
} MoeBatchDesc;

static PyObject* py_launch_diff_q1_moe_batch(PyObject* self, PyObject* args) {
    unsigned long long fn_ptr, stream_ptr;
    unsigned int gx, smem;
    unsigned long long px, pyo;
    int shared_x, H, N, K, GS, P_min, out_is_bf16;
    float tol;
    PyObject* experts_seq;
    if (!PyArg_ParseTuple(
            args, "KKIIKKiiiiifiiO",
            &fn_ptr, &stream_ptr, &gx, &smem, &px, &pyo,
            &shared_x, &H, &N, &K, &GS, &tol, &P_min, &out_is_bf16, &experts_seq)) {
        return NULL;
    }
    if (!ensure_cuda_driver()) {
        PyErr_SetString(PyExc_RuntimeError, "libcuda cuLaunchKernel not available");
        return NULL;
    }
    if (H <= 0 || H > MAX_MOE_BATCH) {
        PyErr_SetString(PyExc_ValueError, "H out of range [1..16]");
        return NULL;
    }
    MoeBatchDesc desc;
    memset(&desc, 0, sizeof(desc));
    PyObject* fast = PySequence_Fast(experts_seq, "experts must be a sequence");
    if (!fast) return NULL;
    if (PySequence_Fast_GET_SIZE(fast) != H) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "experts sequence length mismatch");
        return NULL;
    }
    for (int i = 0; i < H; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(fast, i);
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 9) {
            Py_DECREF(fast);
            PyErr_SetString(PyExc_TypeError, "each expert entry must be a 9-tuple");
            return NULL;
        }
        desc.x_last[i]       = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(item, 0));
        desc.y_last[i]       = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(item, 1));
        desc.packed_lo[i]    = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(item, 2));
        desc.qscale[i]       = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(item, 3));
        desc.qzero[i]        = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(item, 4));
        desc.row_scale[i]    = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(item, 5));
        desc.col_scale[i]    = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(item, 6));
        desc.p_max[i]        = (int)PyLong_AsLong(PyTuple_GET_ITEM(item, 7));
        desc.shared_scale[i] = (int)PyLong_AsLong(PyTuple_GET_ITEM(item, 8));
    }
    Py_DECREF(fast);
    if (PyErr_Occurred()) return NULL;

    void* a_x = (void*)(uintptr_t)px;
    void* a_y = (void*)(uintptr_t)pyo;
    void* kparams[11] = {
        &a_x, &a_y, &desc, &shared_x, &H, &N, &K, &GS, &P_min, &tol, &out_is_bf16
    };
    int rc = g_cuLaunchKernel(
        (void*)(uintptr_t)fn_ptr,
        gx, 1, (unsigned int)H, 256, 1, 1, smem,
        (void*)(uintptr_t)stream_ptr, kparams, NULL
    );
    return PyLong_FromLong((long)rc);
}

/* Kernel #3: pure_mirror_line_step_f32 */
static PyObject* py_launch_line_step(PyObject* self, PyObject* args) {
    unsigned long long fn_ptr, stream_ptr;
    unsigned long long pline, py, ptok, pw, pout;
    int R, H;
    if (!PyArg_ParseTuple(
            args, "KKKKKKKii",
            &fn_ptr, &stream_ptr, &pline, &py, &ptok, &pw, &pout, &R, &H)) {
        return NULL;
    }
    if (!ensure_cuda_driver()) {
        PyErr_SetString(PyExc_RuntimeError, "libcuda cuLaunchKernel not available");
        return NULL;
    }
    void* a0 = (void*)(uintptr_t)pline;
    void* a1 = (void*)(uintptr_t)py;
    void* a2 = (void*)(uintptr_t)ptok;
    void* a3 = (void*)(uintptr_t)pw;
    void* a4 = (void*)(uintptr_t)pout;
    void* kparams[7] = { &a0, &a1, &a2, &a3, &a4, &R, &H };
    int rc = g_cuLaunchKernel(
        (void*)(uintptr_t)fn_ptr,
        (unsigned int)R, 1, 1, 256, 1, 1, 0,
        (void*)(uintptr_t)stream_ptr, kparams, NULL
    );
    return PyLong_FromLong((long)rc);
}

/* Kernel #7: pure_mirror_carve_q1_ballot_f32 */
static PyObject* py_launch_carve_q1_ballot(PyObject* self, PyObject* args) {
    unsigned long long fn_ptr, stream_ptr;
    unsigned long long pu, pxr, prs, ppk, pqs, pqz, pyz, pdy;
    int N, K, root_bits, P_max;
    if (!PyArg_ParseTuple(
            args, "KKKKKKKKKKiiii",
            &fn_ptr, &stream_ptr, &pu, &pxr, &prs, &ppk, &pqs, &pqz, &pyz, &pdy,
            &N, &K, &root_bits, &P_max)) {
        return NULL;
    }
    if (!ensure_cuda_driver()) {
        PyErr_SetString(PyExc_RuntimeError, "libcuda cuLaunchKernel not available");
        return NULL;
    }
    void* a0 = (void*)(uintptr_t)pu;
    void* a1 = (void*)(uintptr_t)pxr;
    void* a2 = (void*)(uintptr_t)prs;
    void* a3 = (void*)(uintptr_t)ppk;
    void* a4 = (void*)(uintptr_t)pqs;
    void* a5 = (void*)(uintptr_t)pqz;
    void* a6 = (void*)(uintptr_t)pyz;
    void* a7 = (void*)(uintptr_t)pdy;
    void* kparams[12] = { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &N, &K, &root_bits, &P_max };
    unsigned int smem = (unsigned int)(8 * (1 + P_max) * (int)sizeof(float));
    int rc = g_cuLaunchKernel(
        (void*)(uintptr_t)fn_ptr,
        (unsigned int)N, 1, 1, 256, 1, 1, smem,
        (void*)(uintptr_t)stream_ptr, kparams, NULL
    );
    return PyLong_FromLong((long)rc);
}

/* Kernel #8: fused multi-iteration L2 Sinkhorn loop in C (0 Python between iterations!) */
static PyObject* py_launch_sinkhorn_iters(PyObject* self, PyObject* args) {
    unsigned long long fn_row, fn_col, stream_ptr;
    unsigned long long pw, pdr, pdc;
    int N, K, iters;
    if (!PyArg_ParseTuple(
            args, "KKKKKKiii",
            &fn_row, &fn_col, &stream_ptr, &pw, &pdr, &pdc, &N, &K, &iters)) {
        return NULL;
    }
    if (!ensure_cuda_driver()) {
        PyErr_SetString(PyExc_RuntimeError, "libcuda cuLaunchKernel not available");
        return NULL;
    }
    void* aw = (void*)(uintptr_t)pw;
    void* adr = (void*)(uintptr_t)pdr;
    void* adc = (void*)(uintptr_t)pdc;
    void* kr[5] = { &aw, &adc, &adr, &N, &K };
    void* kc[5] = { &aw, &adr, &adc, &N, &K };
    unsigned int col_gx = (unsigned int)((K + 31) / 32);
    for (int it = 0; it < iters; ++it) {
        int rc = g_cuLaunchKernel(
            (void*)(uintptr_t)fn_row,
            (unsigned int)N, 1, 1, 256, 1, 1, 0,
            (void*)(uintptr_t)stream_ptr, kr, NULL
        );
        if (rc != 0) return PyLong_FromLong((long)rc);
        rc = g_cuLaunchKernel(
            (void*)(uintptr_t)fn_col,
            col_gx, 1, 1, 32, 8, 1, 0,
            (void*)(uintptr_t)stream_ptr, kc, NULL
        );
        if (rc != 0) return PyLong_FromLong((long)rc);
    }
    return PyLong_FromLong(0);
}

static PyObject* py_launch_sinkhorn_apply(PyObject* self, PyObject* args) {
    unsigned long long fn_app, stream_ptr;
    unsigned long long pw, prs, pcs, peq;
    int N, K;
    if (!PyArg_ParseTuple(
            args, "KKKKKKii",
            &fn_app, &stream_ptr, &pw, &prs, &pcs, &peq, &N, &K)) {
        return NULL;
    }
    if (!ensure_cuda_driver()) {
        PyErr_SetString(PyExc_RuntimeError, "libcuda cuLaunchKernel not available");
        return NULL;
    }
    void* aw = (void*)(uintptr_t)pw;
    void* ars = (void*)(uintptr_t)prs;
    void* acs = (void*)(uintptr_t)pcs;
    void* aeq = (void*)(uintptr_t)peq;
    void* kp[6] = { &aw, &ars, &acs, &aeq, &N, &K };
    int rc = g_cuLaunchKernel(
        (void*)(uintptr_t)fn_app,
        (unsigned int)N, 1, 1, 256, 1, 1, 0,
        (void*)(uintptr_t)stream_ptr, kp, NULL
    );
    return PyLong_FromLong((long)rc);
}

static PyMethodDef QuintCudaFastMethods[] = {
    {"launch_diff_q1_gemv", py_launch_diff_q1_gemv, METH_VARARGS, "Fast C cuLaunchKernel for Kernel #6"},
    {"launch_diff_q1_moe_batch", py_launch_diff_q1_moe_batch, METH_VARARGS, "Fast C batched MoE Kernel #6c"},
    {"launch_line_step", py_launch_line_step, METH_VARARGS, "Fast C cuLaunchKernel for line_step"},
    {"launch_carve_q1_ballot", py_launch_carve_q1_ballot, METH_VARARGS, "Fast C cuLaunchKernel for Ballot Q1 carve"},
    {"launch_sinkhorn_iters", py_launch_sinkhorn_iters, METH_VARARGS, "Fast C loop for L2 Sinkhorn"},
    {"launch_sinkhorn_apply", py_launch_sinkhorn_apply, METH_VARARGS, "Fast C apply for L2 Sinkhorn"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef quint_cuda_fast_module = {
    PyModuleDef_HEAD_INIT,
    "_quint_cuda_fast_ext",
    "Native C CUDA kernel dispatcher for Quintessentia",
    -1,
    QuintCudaFastMethods
};

PyMODINIT_FUNC PyInit__quint_cuda_fast_ext(void) {
    return PyModule_Create(&quint_cuda_fast_module);
}
"""

_LOCK = threading.Lock()
_LOADED = False
_EXT: Optional[Any] = None


def get_fast_cuda_ext() -> Optional[Any]:
    """Return compiled native C CUDA launcher module, or None if compiler unavailable."""
    global _LOADED, _EXT
    if _LOADED:
        return _EXT
    with _LOCK:
        if _LOADED:
            return _EXT
        _LOADED = True
        if os.name == "nt":
            return None
        cc = shutil.which("gcc") or shutil.which("g++")
        inc = sysconfig.get_path("include")
        if not cc or not inc or not os.path.exists(os.path.join(inc, "Python.h")):
            return None
        try:
            cache_dir = Path.home() / ".cache" / "quintessentia" / "native_ext"
            cache_dir.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256(_C_SOURCE.encode("utf-8")).hexdigest()[:16]
            so_path = cache_dir / f"_quint_cuda_fast_ext_{h}.so"
            if not so_path.exists():
                c_path = cache_dir / f"_quint_cuda_fast_ext_{h}.c"
                tmp_so = cache_dir / f"_quint_cuda_fast_ext_{h}.{os.getpid()}.tmp.so"
                c_path.write_text(_C_SOURCE, encoding="utf-8")
                cmd = [
                    cc, "-O3", "-shared", "-fPIC",
                    f"-I{inc}", str(c_path), "-ldl", "-o", str(tmp_so),
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if res.returncode != 0:
                    try:
                        tmp_so.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return None
                os.replace(tmp_so, so_path)
            spec = importlib.util.spec_from_file_location("_quint_cuda_fast_ext", str(so_path))
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _EXT = mod
            return _EXT
        except Exception:
            _EXT = None
            return None
