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
        source = Path(__file__).with_name("mandelbrot.c")
        source_bytes = source.read_bytes()
        build_key = hashlib.sha256(
            source_bytes + platform.machine().encode() + b"-O3-march=native-fopenmp"
        ).hexdigest()[:16]
        cache_dir = Path(tempfile.gettempdir()) / "mandc-native"
        cache_dir.mkdir(exist_ok=True)
        library_path = cache_dir / f"libmandelbrot-{build_key}.so"

        if not library_path.exists():
            temporary_path = library_path.with_name(f"{library_path.name}.{os.getpid()}.tmp")
            subprocess.run(
                [
                    "cc",
                    "-O3",
                    "-march=native",
                    "-fno-math-errno",
                    "-fopenmp",
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
            os.replace(temporary_path, library_path)

        self._library = ctypes.CDLL(str(library_path))
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
