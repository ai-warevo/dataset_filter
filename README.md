# Dataset Face Filter (DDD & SOLID Compliance)

A highly structured, production-grade automated pipeline built following **Domain-Driven Design (DDD)** and **SOLID** principles to clean image datasets by verifying facial embeddings against a master reference profile. 

This tool is explicitly designed to purge unwanted generations, latent anomalies, and out-of-character configurations directly from generation outputs (e.g., ComfyUI / Stable Diffusion pipelines).

---

## 📂 Architecture Overview

The codebase is strictly separated into clean modular layers to ensure zero cyclical complexity, maximum testability, and isolated domain business rules:

```text
dataset_filter/
├── __init__.py
├── domain/                      # 🧠 Pure Domain Logic (Core business layer)
│   ├── __init__.py
│   ├── exceptions.py            # Isolated domain runtime errors
│   ├── models.py                # Value Objects and Domain Entities
│   └── services.py              # Pure Abstract Interfaces 
├── application/                 # ⚙️ Orchestration Layer (Workflow coordination)
│   ├── __init__.py
│   └── purge_service.py         # Coordinates processing routines
└── infrastructure/              # 🔌 Gateways & Extensions (External libraries)
    ├── __init__.py
    └── recognition_engine.py    # Concrete face_recognition implementation
pyproject.toml                   # 📦 Declarative project configuration
main.py                          # 🚀 CLI Bootstrapper (Dependency Injection boundary)
```

---

## 🐳 Containerized Execution (Recommended)

Running the application via Docker isolates the heavy compilation of `dlib` and C++ dependencies completely from your host machine.

### 1. Requirements
Ensure you have [Docker](https://docker.com) and [Docker Compose](https://docker.com) installed.

### 2. Run the Purge Lifecycle
Place your master image and dataset folder in the root directory matching your `docker-compose.yml` declarations, then trigger execution:

```bash
docker compose up --build
```
*Note: Since volumes use active bind mounts, any files purged inside the container will instantly vanish from your local host machine folder.*

---

## 🛠️ Local Prerequisites & Installation (Alternative)

If you prefer to run the application bare-metal without containerization, manage your environment using `uv` for ultra-fast package handling.

### 1. Install `uv`
https://astral.sh


### 2. Install Hardware Compilation Requirements
* **Windows Setup:** Install **Visual Studio Community** and ensure the **"Desktop development with C++"** workload is selected.
* **Linux Setup (Ubuntu/Debian):**
  ```bash
  sudo apt update && sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev
  ```

### 3. Sync Dependencies via `pyproject.toml`
Target your specific environment (e.g., an existing ComfyUI virtual environment) using the `--python` flag to link dependencies safely without overriding existing version pins:

```bash
uv sync
```

---

## 🚀 Native CLI Execution & Parameters

When executing locally without docker wrappers, configure the runtime flags dynamically via the primary entrypoint boundary:

```bash
python main.py --reference "path/to/master_face.png" --input-dir "path/to/dataset" --tolerance 0.55
```

### Argument Parameters

| Long Flag | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `--reference` | **Yes** | *None* | Absolute or relative path to the master ideal reference profile. |
| `--input-dir` | **Yes** | *None* | Path to the workspace target directory containing image assets to prune. |
| `--tolerance` | No | `0.55` | Strictness threshold. Lower values require stricter matching metrics (`0.50`-`0.60` optimal). |
| `--mode` | No | `log` | Operation mode: `log` (dry run), `purge` (delete defects), or `rename` (rename matches only). |

---

## 🔄 Operation Modes

The application behavior is determined by the selected mode. This allows for safe testing before performing destructive operations.

| Mode | Defect Strategy | Match Strategy | Effect |
| :--- | :--- | :--- | :--- |
| `log` (Default) | `LogDefectStrategy` | `NoOpMatchStrategy` | **Dry Run**: Only logs detected defects to the console. No files are modified. |
| `purge` | `PurgeDefectStrategy` | `NoOpMatchStrategy` | **Destructive**: Physically deletes images that do not meet the tolerance threshold. |
| `rename` | `LogDefectStrategy` | `RenameMatchStrategy` | **Transformation**: Logs defects and renames files that successfully match the reference. |


---

## 💎 Linting & Clean Code Standards

This software structure is systematically checked and remains completely compliant with strict automated code inspection configurations:
* **Black**: Enforces precise mathematical code formatting and quote boundaries.
* **Mypy**: Enforces explicit static validation across runtime boundaries.
* **Ruff / Pylint / Flake8**: Enforces clean exception chains, zero nested structural complexity ladders, and optimal logging conventions.
