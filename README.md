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
Ensure you replace token_hf with your actual Hugging Face token.

```bash
docker run --gpus all -it -p 7861:7861 -e SERVER_HOST=0.0.0.0 -e SERVER_PORT=7861 -e HF_TOKEN="token_hf" trellis

```

---

## TRELLIS.2 API Behavior

### `POST /queue` — Submit job

**Data**:

```json
{
  "image": "base64 image",
  "preprocess_image": true,
  "seed": 42,
  "resolution": "1024",
  "ss_guidance_strength": 7.5,
  "ss_guidance_rescale": 0.7,
  "ss_sampling_steps": 12,
  "ss_rescale_t": 5.0,
  "shape_slat_guidance_strength": 7.5,
  "shape_slat_guidance_rescale": 0.5,
  "shape_slat_sampling_steps": 12,
  "shape_slat_rescale_t": 3.0,
  "tex_slat_guidance_strength": 1.0,
  "tex_slat_guidance_rescale": 0.0,
  "tex_slat_sampling_steps": 12,
  "tex_slat_rescale_t": 3.0,
  "decimation_target": 1000000,
  "texture_size": 4096
}
```

| Field | Type | Constraint | Default |
|-------|------|------------|---------|
| `image` | `string` | Base64 valide (PNG, JPEG, WEBP, BMP ou GIF) | **mandatory** |
| `preprocess_image` | `boolean` | — | `true` |
| `seed` | `integer` | — | `42` |
| `resolution` | `string` | `"512"`, `"1024"`, `"1024_cascade"`, `"1536_cascade"` | `"1024"` |
| `ss_guidance_strength` | `number` | `1.0` – `10.0` | `7.5` |
| `ss_guidance_rescale` | `number` | `0.0` – `1.0` | `0.7` |
| `ss_sampling_steps` | `integer` | `1` – `50` | `12` |
| `ss_rescale_t` | `number` | `1.0` – `6.0` | `5.0` |
| `shape_slat_guidance_strength` | `number` | `1.0` – `10.0` | `7.5` |
| `shape_slat_guidance_rescale` | `number` | `0.0` – `1.0` | `0.5` |
| `shape_slat_sampling_steps` | `integer` | `1` – `50` | `12` |
| `shape_slat_rescale_t` | `number` | `1.0` – `6.0` | `3.0` |
| `tex_slat_guidance_strength` | `number` | `1.0` – `10.0` | `1.0` |
| `tex_slat_guidance_rescale` | `number` | `0.0` – `1.0` | `0.0` |
| `tex_slat_sampling_steps` | `integer` | `1` – `50` | `12` |
| `tex_slat_rescale_t` | `number` | `1.0` – `6.0` | `3.0` |
| `decimation_target` | `integer` | `100000` – `1000000` | `1000000` |
| `texture_size` | `integer` | `1024` – `4096` | `4096` |

---

#### Here are **all SSE events** the server can emit after a `POST /queue` request:

#### 1. In queue (waiting)

```json
{
  "status": "in_queue",
  "pos": 0
}
```

- **`status`**: always `"in_queue"`
- **`pos`**: position in the queue (0 = first to be processed)

---

#### 2. Processing

```json
{
  "status": "processing"
}
```

- **`status`**: always `"processing"`
- No other fields.

---

#### 3. Success – Model ready

```json
{
  "status": "ready",
  "uid": "a1b2c3d4e5f6...",
  "timeout": 60.0
}
```

- **`uid`**: unique job identifier, to be used with `GET /uid/{uid}` to download the `.glb` file.
- **`timeout`**: time in seconds before the file is deleted (value of `READY_TIMEOUT`, default `60.0`).

---

#### 4. Failure

```json
{
  "status": "fail",
  "error": "PIPELINE FAILED: ...",
  "timeout": 30.0
}
```

- **`error`**: detailed error message.
- **`timeout`**: time in seconds before cleanup (value of `FAIL_TIMEOUT`, default `30.0`).

---

### Typical event flow

```
data: {"status": "in_queue", "pos": 2}

data: {"status": "in_queue", "pos": 1}

data: {"status": "in_queue", "pos": 0}

data: {"status": "processing"}

data: {"status": "processing"}

data: {"status": "ready", "uid": "abc123...", "timeout": 60.0}
```

Or in case of failure:

```
data: {"status": "in_queue", "pos": 0}

data: {"status": "processing"}

data: {"status": "fail", "error": "PIPELINE FAILED: CUDA out of memory", "timeout": 60.0}
```

---

### `GET /uid/{uid}`

Download the generated `.glb` file using the `uid` received from the SSE events.

| Code | Meaning | Body |
|------|---------|------|
| `200` | File found and returned | Binary `.glb` data (`Content-Type: model/gltf-binary`) |
| `404` | UID not found or expired / job not ready / job failed | Plain text `"Not found"` |