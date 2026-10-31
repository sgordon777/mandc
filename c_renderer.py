"""ctypes wrapper for the native double-precision Mandelbrot renderer."""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class CRenderer:
    """Compile and call the native C Mandelbrot renderer."""

    def __init__(self) -> None:
        self._use_openmp_gpu = os.environ.get("MANDC_OPENMP_GPU") == "1"
        source = Path(__file__).with_name("mandelbrot.c")
        source_bytes = source.read_bytes()
        offload_flags = (
            [
                "-foffload=nvptx-none",
                "-foffload-options=nvptx-none=-fcf-protection=none",
                "-foffload-options=nvptx-none=-fno-stack-protector",
            ]
            if self._use_openmp_gpu
            else ["-foffload=disable"]
        )
        build_key = hashlib.sha256(
            source_bytes
            + platform.machine().encode()
            + b"-O3-march=native-fopenmp"
            + " ".join(offload_flags).encode()
        ).hexdigest()[:16]
        cache_dir = Path(tempfile.gettempdir()) / "mandc-native"
        cache_dir.mkdir(exist_ok=True)
        library_path = cache_dir / f"libmandelbrot-{build_key}.so"

        if not library_path.exists():
            temporary_path = library_path.with_name(f"{library_path.name}.{os.getpid()}.tmp")
            try:
                subprocess.run(
                    [
                        "cc",
                        "-O3",
                        "-march=native",
                        "-fno-math-errno",
                        "-fopenmp",
                        *offload_flags,
                        "-fPIC",
                        "-shared",
                        str(source),
                        "-o",
                        str(temporary_path),
                        "-lm",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as error:
                if not self._use_openmp_gpu:
                    raise
                compiler_output = error.stderr.strip() if error.stderr else str(error)
                raise RuntimeError(
                    "OpenMP NVPTX offload compilation failed. Install the GCC NVPTX "
                    f"offload compiler/runtime.\n{compiler_output}"
                ) from error
            os.replace(temporary_path, library_path)

        self._library = ctypes.CDLL(str(library_path))
        if self._use_openmp_gpu:
            gpu_available = self._library.mandelbrot_openmp_gpu_available
            gpu_available.argtypes = []
            gpu_available.restype = ctypes.c_int
            if not gpu_available():
                raise RuntimeError(
                    "OpenMP GPU offload was requested but no NVIDIA target device is active. "
                    "Check the NVPTX compiler and libgomp offload runtime."
                )
            self._render = self._library.render_mandelbrot_openmp_gpu
        else:
            self._render = self._library.render_mandelbrot_c
        self._render.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint16),
        ]
        self._render.restype = None

    def render(
        self,
        width: int,
        height: int,
        center_real: float,
        center_imag: float,
        span: float,
        max_iterations: int,
        iterations: np.ndarray,
        thread_count: int = 1,
    ) -> np.ndarray:
        """Return per-pixel escape iteration counts with shape (width, height)."""
        self._render(
            width,
            height,
            center_real,
            center_imag,
            span,
            max_iterations,
            thread_count,
            iterations.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        )
        return iterations
