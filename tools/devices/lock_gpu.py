# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
import argparse
import ctypes
import os
import subprocess
import sys
import time
from uuid import UUID


def _cuda_visible_ordinal(gpu_id: int) -> int:
    """Map a physical GPU id to its CUDA-visible logical ordinal.

    With CUDA_VISIBLE_DEVICES set, CUDA renumbers the visible devices from 0,
    so cuDeviceGet needs the ordinal within that list rather than the physical
    id reported by nvidia-smi.
    """
    if gpu_id < 0:
        raise ValueError(f"Invalid physical GPU id {gpu_id}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return gpu_id
    tokens = visible.split(",")
    if not all(token.isascii() and token.isdecimal() for token in tokens):
        raise ValueError(f"Invalid CUDA_VISIBLE_DEVICES={visible!r}")
    ids = [int(token) for token in tokens]
    if gpu_id not in ids:
        raise ValueError(f"GPU {gpu_id} is outside CUDA_VISIBLE_DEVICES={visible!r}")
    return ids.index(gpu_id)


def _lock_gpu_cuda(ordinal: int, expected_uuid: bytes, mem_to_alloc_mb: int) -> bool:
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        if cuda.cuInit(0) != 0:
            return False
        dev = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(dev), ordinal) != 0:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                sys.exit(f"CUDA ordinal {ordinal} is unavailable under the device mask")
            return False
        device_uuid = ctypes.create_string_buffer(16)
        get_uuid = getattr(cuda, "cuDeviceGetUuid_v2", cuda.cuDeviceGetUuid)
        if get_uuid(device_uuid, dev.value) != 0:
            return False
        if device_uuid.raw != expected_uuid:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                sys.exit(f"CUDA ordinal {ordinal} does not match the requested physical GPU")
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


# A HIP context holds no VRAM, so a ROCm lock must allocate more than the idle
# thresholds that runners pass to get_gpu (resource-rocm.sh counts KFD VRAM).
ROCM_MIN_LOCK_MB = 64


def _load_hip() -> ctypes.CDLL:
    candidates: list[str] = []
    try:
        import rocm_sdk  # Pip-installed ROCm SDKs keep the library in a package.

        candidates += [str(path) for path in rocm_sdk.find_libraries("amdhip64")]
    except Exception:
        pass
    candidates += ["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"]
    for root in (os.environ.get("ROCM_PATH"), "/opt/rocm"):
        if root:
            candidates.append(os.path.join(root, "lib", "libamdhip64.so"))
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise OSError("Cannot load the HIP runtime library (libamdhip64.so)")


def lock_gpu_rocm(pci_address: str, mem_to_alloc_mb: int) -> None:
    """Lock the ROCm GPU at ``pci_address`` among the HIP-visible devices."""
    try:
        hip = _load_hip()
        if hip.hipInit(0) != 0:
            sys.exit("hipInit failed")
        count = ctypes.c_int()
        if hip.hipGetDeviceCount(ctypes.byref(count)) != 0:
            sys.exit("hipGetDeviceCount failed")
        target = pci_address.lower()
        for ordinal in range(count.value):
            bus_id = ctypes.create_string_buffer(32)
            if hip.hipDeviceGetPCIBusId(bus_id, len(bus_id), ordinal) != 0:
                continue
            if bus_id.value.decode().lower() == target:
                break
        else:
            sys.exit(f"No HIP-visible device has PCI address {pci_address}")
        if hip.hipSetDevice(ordinal) != 0:
            sys.exit(f"hipSetDevice({ordinal}) failed")
        size_bytes = max(mem_to_alloc_mb, ROCM_MIN_LOCK_MB) * 1024 * 1024
        pointer = ctypes.c_void_p()
        if hip.hipMalloc(ctypes.byref(pointer), ctypes.c_size_t(size_bytes)) != 0:
            sys.exit(f"hipMalloc failed on {pci_address}")
    except OSError as exc:
        sys.exit(f"Cannot lock ROCm GPU {pci_address}: {exc}")
    print("READY", flush=True)
    time.sleep(31536000)


def lock_gpu(gpu_id: int, mem_to_alloc_mb: int):
    try:
        ordinal = _cuda_visible_ordinal(gpu_id)
        result = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
        )
        expected_uuid = UUID(result.strip().removeprefix("GPU-")).bytes
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Cannot identify GPU {gpu_id}: {exc}")
    if _lock_gpu_cuda(ordinal, expected_uuid, mem_to_alloc_mb):
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

        # EGL has its own enumeration order. Match EGL_DEVICE_UUID_EXT before
        # creating a display or allocating any memory on the device.
        addr = eglGetProcAddress(b"eglQueryDeviceBinaryEXT")
        if not addr:
            raise RuntimeError("EGL cannot verify device UUIDs")
        query_binary = ctypes.CFUNCTYPE(
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        )(addr)
        for device in devices:
            value = ctypes.create_string_buffer(16)
            size = ctypes.c_int()
            if (
                query_binary(device, 0x335C, 16, value, ctypes.byref(size))
                and size.value == 16
                and value.raw == expected_uuid
            ):
                break
        else:
            raise RuntimeError(f"No EGL device matches GPU {gpu_id}")
        display = eglGetPlatformDisplayEXT(0x313F, device, None)  # EGL_PLATFORM_DEVICE_EXT
        if not display:
            raise Exception("eglGetPlatformDisplayEXT failed")

        # EGL handles are pointers, including on 64-bit headless nodes.
        display = ctypes.c_void_p(display)
        if egl.eglInitialize(display, None, None) == 0:
            raise Exception("eglInitialize failed")
        if egl.eglBindAPI(0x30A2) == 0:
            raise Exception("eglBindAPI failed")  # EGL_OPENGL_API

        config_attribs = (ctypes.c_int * 7)(
            0x3033,
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
            or not num_configs.value
        ):
            raise Exception("eglChooseConfig failed")

        egl.eglCreateContext.restype = ctypes.c_void_p
        ctx = ctypes.c_void_p(egl.eglCreateContext(display, config, None, None))
        if not ctx:
            raise Exception("eglCreateContext failed")

        pbuffer_attribs = (ctypes.c_int * 5)(0x3057, 1, 0x3056, 1, 0x3038)  # WIDTH, HEIGHT
        egl.eglCreatePbufferSurface.restype = ctypes.c_void_p
        surface = ctypes.c_void_p(egl.eglCreatePbufferSurface(display, config, pbuffer_attribs))
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
        if not buf.value or gl.glGetError() != 0:
            raise RuntimeError("OpenGL buffer allocation failed")

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
    parser.add_argument(
        "--rocm-pci-address",
        help="Lock the ROCm GPU at this PCI address (resource.sh resolves it from gpu_id).",
    )
    args = parser.parse_args()
    if args.rocm_pci_address:
        lock_gpu_rocm(args.rocm_pci_address, args.mem)
    else:
        lock_gpu(args.gpu_id, args.mem)
