# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
import argparse
import ctypes
import os
import sys
import time


def _cuda_visible_ordinal(gpu_id: int) -> int:
    """Map a physical GPU id to its CUDA-visible logical ordinal.

    With CUDA_VISIBLE_DEVICES set, CUDA renumbers the visible devices from 0,
    so cuDeviceGet needs the ordinal within that list rather than the physical
    id reported by nvidia-smi.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        return gpu_id
    ids = [token.strip() for token in visible.split(",")]
    try:
        return ids.index(str(gpu_id))
    except ValueError:
        return gpu_id


def _lock_gpu_cuda(gpu_id: int, mem_to_alloc_mb: int) -> bool:
    try:
        ordinal = _cuda_visible_ordinal(gpu_id)
        cuda = ctypes.CDLL("libcuda.so.1")
        if cuda.cuInit(0) != 0:
            return False
        dev = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(dev), ordinal) != 0:
            return False
        ctx = ctypes.c_void_p()
        if cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev.value) != 0:
            return False
        dptr = ctypes.c_uint64()
        size_bytes = mem_to_alloc_mb * 1024 * 1024
        if cuda.cuMemAlloc_v2(ctypes.byref(dptr), size_bytes) != 0:
            return False
        return True
    except Exception:
        return False


def lock_gpu(gpu_id: int, mem_to_alloc_mb: int):
    # First try CUDA driver API (most robust on headless GPU nodes)
    if _lock_gpu_cuda(gpu_id, mem_to_alloc_mb):
        print("READY", flush=True)
        time.sleep(31536000)
        return

    # Fallback to EGL implementation
    try:
        if "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
            standard_vendor_files = [
                "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
                "/etc/glvnd/egl_vendor.d/10_nvidia.json",
            ]
            for vendor_file in standard_vendor_files:
                if os.path.exists(vendor_file):
                    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = vendor_file
                    break
            else:
                cfg_path = "/tmp/10_nvidia_auto.json"
                if not os.path.exists(cfg_path):
                    try:
                        with open(cfg_path, "w") as f:
                            f.write(
                                '{"file_format_version":"1.0.0",'
                                '"ICD":{"library_path":"libEGL_nvidia.so.0"}}'
                            )
                    except Exception:
                        pass
                if os.path.exists(cfg_path):
                    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = cfg_path

        egl = ctypes.CDLL("libEGL.so.1")
        gl = ctypes.CDLL("libGL.so.1")

        eglGetProcAddress = egl.eglGetProcAddress
        eglGetProcAddress.restype = ctypes.c_void_p
        eglGetProcAddress.argtypes = [ctypes.c_char_p]

        addr = eglGetProcAddress(b"eglQueryDevicesEXT")
        if not addr:
            raise Exception("eglQueryDevicesEXT not found")
        eglQueryDevicesEXT = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        )(addr)

        addr = eglGetProcAddress(b"eglGetPlatformDisplayEXT")
        if not addr:
            raise Exception("eglGetPlatformDisplayEXT not found")
        eglGetPlatformDisplayEXT = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
        )(addr)

        num_devices = ctypes.c_int()
        if eglQueryDevicesEXT(0, None, ctypes.byref(num_devices)) == 0:
            raise Exception("eglQueryDevicesEXT failed")
        if num_devices.value == 0:
            raise Exception("No EGL devices found")

        devices = (ctypes.c_void_p * num_devices.value)()
        if eglQueryDevicesEXT(num_devices.value, devices, ctypes.byref(num_devices)) == 0:
            raise Exception("eglQueryDevicesEXT failed")

        if gpu_id >= num_devices.value:
            raise Exception(f"GPU ID {gpu_id} >= {num_devices.value} EGL devices")

        display = eglGetPlatformDisplayEXT(0x313F, devices[gpu_id], None)  # EGL_PLATFORM_DEVICE_EXT
        if not display:
            raise Exception("eglGetPlatformDisplayEXT failed")

        if egl.eglInitialize(display, None, None) == 0:
            raise Exception("eglInitialize failed")
        if egl.eglBindAPI(0x30A2) == 0:
            raise Exception("eglBindAPI failed")  # EGL_OPENGL_API

        config_attribs = (ctypes.c_int * 7)(
            0x3028,
            0x0001,  # EGL_SURFACE_TYPE, EGL_PBUFFER_BIT
            0x3040,
            0x0008,  # EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT
            0x3038,  # EGL_NONE
        )
        config = ctypes.c_void_p()
        num_configs = ctypes.c_int()
        if (
            egl.eglChooseConfig(
                display,
                config_attribs,
                ctypes.byref(config),
                1,
                ctypes.byref(num_configs),
            )
            == 0
        ):
            raise Exception("eglChooseConfig failed")

        ctx = egl.eglCreateContext(display, config, 0, None)
        if not ctx:
            raise Exception("eglCreateContext failed")

        pbuffer_attribs = (ctypes.c_int * 5)(0x3057, 1, 0x3056, 1, 0x3038)  # WIDTH, HEIGHT
        surface = egl.eglCreatePbufferSurface(display, config, pbuffer_attribs)
        if not surface:
            raise Exception("eglCreatePbufferSurface failed")

        if egl.eglMakeCurrent(display, surface, surface, ctx) == 0:
            raise Exception("eglMakeCurrent failed")

        buf = ctypes.c_uint()
        gl.glGenBuffers(1, ctypes.byref(buf))
        gl.glBindBuffer(0x8892, buf.value)  # GL_ARRAY_BUFFER

        gl.glBufferData.argtypes = [
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        size = ctypes.c_size_t(mem_to_alloc_mb * 1024 * 1024)
        gl.glBufferData(0x8892, size, None, 0x88E4)  # GL_STATIC_DRAW

        # Indicate we're ready after allocation succeeds
        print("READY", flush=True)

        time.sleep(31536000)  # Sleep for a year, waiting to be killed.
    except Exception as e:
        print(f"ERROR: Failed to lock GPU {gpu_id}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lock a GPU by allocating some VRAM.")
    parser.add_argument("gpu_id", type=int, help="ID of the GPU to lock.")
    parser.add_argument("--mem", type=int, default=4, help="Amount of memory in MB to allocate.")
    args = parser.parse_args()
    lock_gpu(args.gpu_id, args.mem)
