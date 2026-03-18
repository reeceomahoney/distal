#!/bin/bash
# Diagnostic script for CUDA + EGL performance on cloud GPU instances
# Run on each instance and save output: bash diagnose_gpu.sh > diag_$(hostname).txt 2>&1

set -e

echo "=== TIMESTAMP ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

echo -e "\n=== HOSTNAME / INSTANCE ID ==="
hostname
cat /etc/hostname 2>/dev/null || true
# Cloud provider metadata
curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true
curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || true
curl -s --connect-timeout 2 -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/machine-type 2>/dev/null || true

echo -e "\n=== OS / KERNEL ==="
uname -a
cat /etc/os-release 2>/dev/null | head -5

echo -e "\n=== CPU ==="
lscpu | grep -E "Model name|Socket|Core|Thread|CPU\(s\)|MHz|cache|NUMA|Architecture"
echo "--- CPU governor ---"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "N/A"

echo -e "\n=== MEMORY ==="
free -h
echo "--- NUMA topology ---"
numactl --hardware 2>/dev/null || echo "numactl not available"

echo -e "\n=== NVIDIA DRIVER ==="
cat /proc/driver/nvidia/version 2>/dev/null || echo "N/A"

echo -e "\n=== NVIDIA-SMI ==="
nvidia-smi
echo "--- Clocks and power ---"
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE 2>/dev/null | head -80
echo "--- PCIe link ---"
nvidia-smi -q -d PCIE 2>/dev/null | grep -E "Link|Gen|Width" || true
echo "--- ECC ---"
nvidia-smi -q -d ECC 2>/dev/null | grep -E "Pending|Current|Volatile|Aggregate" | head -20 || true
echo "--- Retired pages ---"
nvidia-smi -q -d RETIRED_PAGES 2>/dev/null | head -20 || true
echo "--- GPU topology ---"
nvidia-smi topo -m 2>/dev/null || true

echo -e "\n=== CUDA VERSION ==="
nvcc --version 2>/dev/null || echo "nvcc not found"
echo "--- libcuda ---"
ls -la /usr/lib/x86_64-linux-gnu/libcuda* 2>/dev/null || ls -la /usr/lib64/libcuda* 2>/dev/null || true
echo "--- CUDA_VISIBLE_DEVICES ---"
echo "${CUDA_VISIBLE_DEVICES:-not set}"

echo -e "\n=== EGL / DISPLAY ==="
echo "DISPLAY=${DISPLAY:-not set}"
echo "EGL_DEVICE_ID=${EGL_DEVICE_ID:-not set}"
echo "MUJOCO_GL=${MUJOCO_GL:-not set}"
echo "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-not set}"
echo "--- EGL libraries ---"
ls -la /usr/lib/x86_64-linux-gnu/libEGL* 2>/dev/null || ls -la /usr/lib64/libEGL* 2>/dev/null || true
ls -la /usr/lib/x86_64-linux-gnu/libGLESv2* 2>/dev/null || true
echo "--- EGL vendor ICDs ---"
ls -la /usr/share/glvnd/egl_vendor.d/ 2>/dev/null || true
cat /usr/share/glvnd/egl_vendor.d/*.json 2>/dev/null || true
echo "--- nvidia EGL icd ---"
find /usr -name "*egl*nvidia*" -o -name "*nvidia*egl*" 2>/dev/null || true
echo "--- libOpenGL / libGL ---"
ls -la /usr/lib/x86_64-linux-gnu/libOpenGL* 2>/dev/null || true
ls -la /usr/lib/x86_64-linux-gnu/libGL.so* 2>/dev/null || true
echo "--- Mesa vs NVIDIA GL ---"
ldconfig -p 2>/dev/null | grep -E "libEGL|libGL|libOpenGL" || true

echo -e "\n=== VULKAN (may affect EGL) ==="
ls /usr/share/vulkan/icd.d/ 2>/dev/null || true
vulkaninfo --summary 2>/dev/null | head -20 || echo "vulkaninfo not available"

echo -e "\n=== PCIe TOPOLOGY ==="
lspci -tv 2>/dev/null | head -40 || true
echo "--- GPU PCIe details ---"
lspci -vvs $(lspci | grep -i nvidia | head -1 | awk '{print $1}') 2>/dev/null | grep -E "LnkSta|LnkCap|NUMA|Width|Speed" || true

echo -e "\n=== IOMMU / VFIO ==="
dmesg 2>/dev/null | grep -i -E "iommu|vfio|passthrough" | tail -10 || true

echo -e "\n=== DISK ==="
df -h / /tmp
echo "--- I/O scheduler ---"
cat /sys/block/$(lsblk -nd --output NAME | head -1)/queue/scheduler 2>/dev/null || true
echo "--- Storage type ---"
lsblk -d -o NAME,ROTA,SIZE,MODEL 2>/dev/null || true

echo -e "\n=== NETWORK ==="
ip link show 2>/dev/null | grep -E "state UP|mtu" || true
echo "--- Bandwidth (to check for noisy neighbor) ---"
ethtool $(ip route get 1.1.1.1 2>/dev/null | awk '{print $5; exit}') 2>/dev/null | grep Speed || true

echo -e "\n=== CONTAINER / VIRTUALIZATION ==="
echo "--- In container? ---"
cat /proc/1/cgroup 2>/dev/null | head -5 || true
cat /run/systemd/container 2>/dev/null || echo "not in systemd container"
echo "--- Hypervisor ---"
systemd-detect-virt 2>/dev/null || echo "N/A"
dmesg 2>/dev/null | grep -i hypervisor | head -3 || true
echo "--- Singularity ---"
echo "SINGULARITY_CONTAINER=${SINGULARITY_CONTAINER:-not set}"

echo -e "\n=== NVIDIA PERSISTENCE / COMPUTE MODE ==="
nvidia-smi -q -d COMPUTE 2>/dev/null | grep -E "Compute Mode|Persistence" || true
echo "--- nvidia-persistenced ---"
pgrep -a nvidia-persist 2>/dev/null || echo "not running"

echo -e "\n=== PROCESS CONTENTION ==="
echo "--- Other GPU processes ---"
nvidia-smi pmon -c 1 2>/dev/null || true
echo "--- System load ---"
uptime
echo "--- Top CPU consumers ---"
ps aux --sort=-%cpu | head -10

echo -e "\n=== KERNEL MODULES ==="
lsmod | grep -i -E "nvidia|nouveau|egl|drm" || true

echo -e "\n=== DMESG GPU ERRORS ==="
dmesg 2>/dev/null | grep -i -E "nvidia|gpu|xid|nvrm|pci.*error|AER" | tail -20 || true

echo -e "\n=== QUICK EGL PROBE ==="
python3 -c "
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
try:
    from OpenGL import EGL
    display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    major, minor = EGL.EGLint(), EGL.EGLint()
    ok = EGL.eglInitialize(display, major, minor)
    print(f'EGL init: {ok}, version: {major.value}.{minor.value}')
    vendor = EGL.eglQueryString(display, EGL.EGL_VENDOR)
    print(f'EGL vendor: {vendor}')
    EGL.eglTerminate(display)
except Exception as e:
    print(f'EGL probe failed: {e}')
" 2>&1 || echo "EGL probe skipped"

echo -e "\n=== QUICK GPU BENCHMARK ==="
python3 -c "
import time, subprocess
try:
    import torch
    if torch.cuda.is_available():
        d = torch.device('cuda')
        print(f'PyTorch CUDA: {torch.version.cuda}')
        print(f'Device: {torch.cuda.get_device_name(0)}')
        print(f'Capability: {torch.cuda.get_device_capability(0)}')
        # Memory bandwidth test
        x = torch.randn(8192, 8192, device=d)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            y = x @ x
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f'Matmul 8192x8192 x20: {dt:.3f}s')
        del x, y
        # Small CUDA+CPU mixed (simulates EGL interop overhead)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            a = torch.randn(256, 256, device=d)
            b = a.cpu()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f'CUDA->CPU transfer 256x256 x200: {dt:.3f}s')
        print(f'GPU memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB used, {torch.cuda.get_device_properties(0).total_mem/1e9:.2f} GB total')
except Exception as e:
    print(f'GPU benchmark failed: {e}')
" 2>&1 || echo "Benchmark skipped"

echo -e "\n=== MUJOCO EGL RENDER BENCHMARK ==="
python3 -c "
import time, os
os.environ.setdefault('MUJOCO_GL', 'egl')
try:
    import mujoco
    xml = '<mujoco><worldbody><light pos=\"0 0 3\"/><body><geom type=\"sphere\" size=\"0.1\"/></body></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    # warmup
    for _ in range(10):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        renderer.render()
    t0 = time.perf_counter()
    for _ in range(500):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        img = renderer.render()
    dt = time.perf_counter() - t0
    print(f'MuJoCo EGL render 500 frames: {dt:.3f}s ({500/dt:.1f} FPS)')
    renderer.close()
except Exception as e:
    print(f'MuJoCo benchmark failed: {e}')
" 2>&1 || echo "MuJoCo benchmark skipped"

echo -e "\n=== DONE ==="
