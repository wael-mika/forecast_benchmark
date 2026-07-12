# GPU Setup — CUDA PyTorch on Windows

## Your hardware

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 SUPER |
| VRAM | 8.6 GB |
| CUDA driver version | 12.6 (NVIDIA-SMI 560.94) |
| OS | Windows 10 Education |

---

## What went wrong (and why)

### Problem 1 — Wrong CUDA wheel index in `requirements.txt`

The original `requirements.txt` pointed to the `cu121` (CUDA 12.1) wheel index:

```
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu121
```

**Why this matters:** PyTorch CUDA wheels are compiled against a specific CUDA toolkit
version. The `cu121` index only had wheels up to torch `2.5.1`. Torch `2.10.0` was not
available there — only `2.10.0+cu126` existed (on the `cu126` index), matching our driver.

---

### Problem 2 — `pip install -e .` silently overwrote the CUDA build

After installing `torch==2.5.1+cu121` successfully, running:

```bash
pip install -e ".[dev]"
```

caused pip to resolve `torch>=2.10` (from `pyproject.toml`) against the **default PyPI index**,
which does have `torch==2.10.0` — but as a **CPU-only wheel**. Pip then silently replaced the
working CUDA build with the CPU version. Result:

```
CUDA available: False   ← even though the GPU was there all along
```

This is a subtle but common trap: pip always re-resolves all dependencies during an editable
install, and without specifying the CUDA index it falls back to PyPI's CPU wheel.

---

## The solution (step-by-step)

### 1. Create a venv with Python 3.11

```bash
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

> **Why Python 3.11?** `pyproject.toml` requires `python >= 3.10`. The system default
> (`python3`) was Python 3.9 — too old. Python 3.11 was installed and available via `py -3.11`.

---

### 2. Install all non-torch dependencies first

```bash
.venv\Scripts\python.exe -m pip install numpy==2.4.3 pandas==3.0.1 scipy==1.17.1 \
    pyarrow==23.0.1 xgboost==3.2.0 matplotlib==3.10.8 pillow==12.1.1 \
    PyYAML==6.0.3 pytest==9.0.2
```

---

### 3. Install PyTorch with the correct CUDA wheel — BEFORE the editable install

```bash
.venv\Scripts\python.exe -m pip install "torch==2.10.0+cu126" \
    --index-url https://download.pytorch.org/whl/cu126
```

> **Key rule:** always install torch from the CUDA wheel index **before** running
> `pip install -e .`. If you run the editable install first, pip will pull the CPU wheel
> from PyPI and you must force-reinstall:
>
> ```bash
> pip install "torch==2.10.0+cu126" \
>     --index-url https://download.pytorch.org/whl/cu126 \
>     --force-reinstall --no-deps
> ```

---

### 4. Install the project in editable mode

```bash
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Because torch is already installed and satisfies `torch>=2.10`, pip does not touch it here.

---

### 5. Verify GPU is visible

```bash
.venv\Scripts\python.exe -c "
import torch
print(torch.__version__)           # 2.10.0+cu126
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 2060 SUPER
"
```

---

## How to find the right wheel index for your GPU

1. Run `nvidia-smi` — the top-right corner shows **CUDA Version** (e.g. `12.6`).
2. Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and
   select your OS / CUDA version to get the exact install command.
3. As a rule of thumb:

| Driver CUDA version | Use index |
|---|---|
| 12.6+ | `https://download.pytorch.org/whl/cu126` |
| 12.1–12.4 | `https://download.pytorch.org/whl/cu121` |
| 11.8 | `https://download.pytorch.org/whl/cu118` |
| No GPU | `https://download.pytorch.org/whl/cpu` |

> CUDA wheels are **forward-compatible within minor versions**: a `cu121` wheel runs fine
> on a machine with CUDA 12.6 drivers. However, the newest PyTorch versions may only be
> published on the latest index (`cu126`), so always prefer the matching version.

---

## Preventing sleep during long training runs

Training all models takes several hours. Windows will sleep and kill the process unless
you prevent it. Since changing power settings requires Administrator rights, the safest
no-admin workaround is a hidden PowerShell keepalive:

```powershell
Start-Process powershell -ArgumentList '-WindowStyle Hidden -Command "
  $ws = New-Object -ComObject WScript.Shell
  while($true) { $ws.SendKeys(''{F15}''); Start-Sleep -Seconds 55 }
"' -WindowStyle Hidden
```

This presses F15 (a key with no visible effect) every 55 seconds, which resets the
inactivity timer. To stop it when training is done, open Task Manager and end the
hidden `powershell.exe` process.
