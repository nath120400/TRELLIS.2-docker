# TRELLIS.2 Docker API

[![TRELLIS.2 - GitHub](https://img.shields.io/badge/TRELLIS.2-GitHub-181717?logo=github)](https://github.com/microsoft/TRELLIS.2) [![TRELLIS.2 - HuggingFace](https://img.shields.io/badge/TRELLIS.2_4B-HuggingFace-FFD21E?logo=huggingface)](https://huggingface.co/microsoft/TRELLIS.2-4B)

> [!IMPORTANT]  
> <p align="justify">🚧 This project currently only supports CUDA.</p> 

> [!NOTE]  
> <p align="justify">`RTX 3090 - WSL Ubuntu-24.04 w/ 1536_cascade -> 🟢`</p> 

> [!CAUTION]
> <p align="justify">You must request and be granted access to specific gated models on Hugging Face before running the container. If you do not have access, the execution will fail.</p> 

## 📋 Prerequisites

Before starting, ensure you have a Hugging Face account and have been granted access to the following gated models:
* [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
* [DINOv3](https://huggingface.co/collections/facebook/dinov3)

You will also need your Hugging Face Access Token (`HF_TOKEN`).

## 🚀 Installation & Usage

**1. Clone the repository and initialize submodules**
```bash
git clone git@github.com:nath120400/TRELLIS.2-docker.git
cd TRELLIS.2-docker/
git submodule update --init --recursive
```

2. Build the Docker image

```bash
docker build -t trellis .
```

3. Run the container
Ensure you replace le token hf with your actual Hugging Face token.

```bash
docker run --gpus all -it -p 7861:7861 -e HF_TOKEN="le token hf"

```

---

### 🔄 API States (`state["status"]`)

The server uses a simple global state to track the background 3D generation process. It cycles through 4 main states:

* **`IDLE`**: Default state. The server is ready and waiting for a new image.
* **`PROCESSING`**: A generation task is currently running. The server is locked and will reject new `POST` requests until the current job finishes.
* **`DONE`**: Generation was successful. The `.glb` file is temporarily saved and ready to be downloaded.
* **`ERROR`**: The generation crashed (e.g., pipeline failure, VRAM OOM, or export issue). The error message is stored to be fetched by the user.

---

### ⚠️ HTTP Responses & Errors

#### `POST /` (Start Generation)
* 🟢 **`200 OK`**: Image received and generation started in the background.
* 🟡 **`102 Processing`**: Cannot start a new job. Another generation is already in progress.
* 🔴 **`400 Bad Request`**: Invalid JSON payload or the base64 image could not be decoded.

#### `GET /` (Check Status / Retrieve Result)
* 🟢 **`200 OK`**: Returns the `.glb` file (`model/gltf-binary`). Automatically deletes the temp file and resets the server to `IDLE`.
* 🟡 **`102 Processing`**: The server is still computing the 3D model. Keep polling.
* 🔴 **`404 Not Found`**: The server is `IDLE` (no job has been started) or the file went missing after completion.
* 🔴 **`500 Internal Server Error`**: The generation failed. Returns the exact crash message (`Crash: ...`) and resets the server to `IDLE`.