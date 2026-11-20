# PoseGuard AI — Installation & Setup Guide (Windows CMD)

This document provides exact, tested step-by-step instructions to set up, install, verify, and run PoseGuard AI on Windows 10 Pro 64-bit (or later) without requiring an NVIDIA GPU or external cloud infrastructure.

---

## 1. System Requirements

- **Operating System**: Windows 10 Pro / Home / Enterprise (64-bit).
- **Processor**: Dual-core or Quad-core x86_64 CPU (Tested on AMD PRO A12-9800B R7, 4 Cores).
- **Memory**: 8 GB RAM (PoseGuard AI operates comfortably with $< 300$ MB peak memory usage).
- **Storage**: ~800 MB free space (Virtual environment, PyTorch CPU wheel, models, and datasets).
- **Python Version**: Python 3.12 (preferred) or Python 3.10+.
- **Graphics / CUDA**: None required. 100% CPU-compatible execution.
- **Budget**: ₹0. All dependencies and models are free and open-source.

---

## 2. Step-by-Step Installation

Open the standard Windows Command Prompt (`cmd.exe`) and execute the following commands:

### Step 1: Navigate to the Project Root
```cmd
cd poseguard-ai
```

### Step 2: Create a Project-Local Virtual Environment
```cmd
py -3.12 -m venv .venv
```
*(If `py -3.12` is not installed, use `python -m venv .venv`)*

### Step 3: Activate the Virtual Environment
```cmd
call .venv\Scripts\activate.bat
```
*(Your prompt will now display `(.venv)`)*

### Step 4: Upgrade Packaging Tools
```cmd
python -m pip install --upgrade pip setuptools wheel
```

### Step 5: Install PyTorch (CPU-Only Build)
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 6: Install Core Dependencies
```cmd
pip install numpy scipy scikit-learn fastapi "uvicorn[standard]" pydantic pytest matplotlib requests psutil python-multipart httpx
```

### Step 7: Install PoseGuard Package in Editable Mode
```cmd
pip install -e .
```

---

## 3. Installation Verification

Verify that PoseGuard AI is correctly installed and all subsystems are operational:

### A. Check Package Version & Model Status
```cmd
poseguard --version
poseguard info
```

Expected output:
```
PoseGuard AI 1.0.0 (Model: poseguard-tiny-v1)
============================================================
  POSEGUARD AI SYSTEM INFO
============================================================
Software Version: 1.0.0
Model Version: poseguard-tiny-v1
Model File: models\poseguard_v1.pt
...
```

### B. Run the Automated Test Suite (26 Tests)
```cmd
pytest tests/ -v
```

Expected result:
```
================== 26 passed in 76.5s ==================
```

---

## 4. Running the Application

### Launch the Desktop Web Application
```cmd
poseguard serve --port 8000
```
Open your browser to:
👉 **`http://127.0.0.1:8000`**

### Run Analysis via CLI
```cmd
poseguard analyze data\raw\subj_01_walk_clean.bvh
poseguard analyze data\synthetic\subj_01_walk_foot_sliding.npz
```

---

## 5. Troubleshooting & FAQ

### Issue: `'py' is not recognized as an internal or external command`
**Fix**: Ensure Python is installed and check "Add Python to PATH" during installation. Alternatively, use:
```cmd
"C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
```

### Issue: `RuntimeError: Form data requires python-multipart`
**Fix**: Run:
```cmd
pip install python-multipart
```

### Issue: Port 8000 is already in use
**Fix**: Launch on a custom port:
```cmd
poseguard serve --port 8080
```

