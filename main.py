import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import base64
import io
import json
import asyncio
import uuid
import tempfile
import time

from aiohttp import web
from PIL import Image

import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# -----------------------------------------------------------------------------
# ENV / CONFIG
# -----------------------------------------------------------------------------
HOST          = os.getenv("SERVER_HOST", "0.0.0.0")
PORT          = int(os.getenv("SERVER_PORT", "7861"))
FAIL_TIMEOUT  = float(os.getenv("FAIL_TIMEOUT", 60.0))
READY_TIMEOUT = float(os.getenv("READY_TIMEOUT", 60.0))
MAX_QUEUE     = int(os.getenv("MAX_QUEUE", 16))

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
pipeline.cuda()

# -----------------------------------------------------------------------------
# PIPELINE
# -----------------------------------------------------------------------------
def img2glb(
    image: Image.Image,
    output_path: str,
    preprocess_image: bool = True,
    seed: int = 42,
    resolution: str = "1024",
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
    decimation_target: int = 1_000_000,
    texture_size: int = 4096,
) -> tuple:
    if resolution not in ('512', '1024', '1024_cascade', '1536_cascade'):
        return False, "WRONG RESOLUTION: accepted values are ['512', '1024', '1024_cascade', '1536_cascade']"
    try:
        image = image.convert("RGBA")
        outputs, latents = pipeline.run(
            image,
            seed=seed,
            preprocess_image=preprocess_image,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale,
                "rescale_t": ss_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": shape_slat_sampling_steps,
                "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale,
                "rescale_t": shape_slat_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": tex_slat_sampling_steps,
                "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale,
                "rescale_t": tex_slat_rescale_t,
            },
            pipeline_type=resolution,
            return_latent=True,
        )
    except Exception as e:
        return False, f"PIPELINE FAILED: {e}"
    try:
        mesh = outputs[0]
        mesh.simplify(16777216)
        _, _, res = latents
        torch.cuda.empty_cache()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0
        )
    except Exception as e:
        return False, f"EXTRACTION FAILED: {e}"
    try:
        glb.export(output_path, extension_webp=True)
        torch.cuda.empty_cache()
        return True, output_path
    except Exception as e:
        return False, f"EXPORT FAILED: {e}"

# -----------------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------------
def validate_payload(data: dict):
    errs = []
    img = None
    b64 = data.get("image")
    if not isinstance(b64, str):
        errs.append("image: must be a base64-encoded string")
    else:
        try:
            raw = base64.b64decode(b64, validate=True)
            img = Image.open(io.BytesIO(raw))
            if img.format not in {"PNG", "JPEG", "JPG", "WEBP", "BMP", "GIF"}:
                errs.append(f"image: unsupported format '{img.format}'. Accepted: PNG, JPEG, WebP, BMP, GIF")
        except Exception as e:
            errs.append(f"image: unreadable or not valid base64 ({e})")

    def get_bool(k, d):
        v = data.get(k, d)
        if not isinstance(v, bool):
            errs.append(f"{k}: must be a boolean")
            return d
        return v

    def get_int(k, d, lo=None, hi=None):
        v = data.get(k, d)
        if isinstance(v, bool) or not isinstance(v, int):
            errs.append(f"{k}: must be an integer")
            return d
        if lo is not None and v < lo: errs.append(f"{k}: must be >= {lo}")
        if hi is not None and v > hi: errs.append(f"{k}: must be <= {hi}")
        return v

    def get_float(k, d, lo=None, hi=None):
        v = data.get(k, d)
        if not isinstance(v, (int, float)):
            errs.append(f"{k}: must be a number")
            return d
        v = float(v)
        if lo is not None and v < lo: errs.append(f"{k}: must be >= {lo}")
        if hi is not None and v > hi: errs.append(f"{k}: must be <= {hi}")
        return v

    def get_choice(k, d, choices):
        v = data.get(k, d)
        if v not in choices:
            errs.append(f"{k}: must be one of {choices}")
            return d
        return v

    params = {
        "preprocess_image":             get_bool("preprocess_image", True),
        "seed":                         get_int("seed", 42),
        "resolution":                   get_choice("resolution", "1024", ('512','1024','1024_cascade','1536_cascade')),

        "ss_guidance_strength":         get_float("ss_guidance_strength", 7.5, 1.0, 10.0),
        "ss_guidance_rescale":          get_float("ss_guidance_rescale", 0.7, 0.0, 1.0),
        "ss_sampling_steps":            get_int("ss_sampling_steps", 12, 1, 50),
        "ss_rescale_t":                 get_float("ss_rescale_t", 5.0, 1.0, 6.0),

        "shape_slat_guidance_strength": get_float("shape_slat_guidance_strength", 7.5, 1.0, 10.0),
        "shape_slat_guidance_rescale":  get_float("shape_slat_guidance_rescale", 0.5, 0.0, 1.0),
        "shape_slat_sampling_steps":    get_int("shape_slat_sampling_steps", 12, 1, 50),
        "shape_slat_rescale_t":         get_float("shape_slat_rescale_t", 3.0, 1.0, 6.0),

        "tex_slat_guidance_strength":   get_float("tex_slat_guidance_strength", 1.0, 1.0, 10.0),
        "tex_slat_guidance_rescale":    get_float("tex_slat_guidance_rescale", 0.0, 0.0, 1.0),
        "tex_slat_sampling_steps":      get_int("tex_slat_sampling_steps", 12, 1, 50),
        "tex_slat_rescale_t":           get_float("tex_slat_rescale_t", 3.0, 1.0, 6.0),

        "decimation_target":            get_int("decimation_target", 1_000_000, 100_000, 1_000_000),
        "texture_size":                 get_int("texture_size", 4096, 1024, 4096),
    }
    if "output_path" in data and not isinstance(data["output_path"], (str, type(None))):
        errs.append("output_path: must be a string or null")

    if errs:
        return None, None, "; ".join(errs)
    return img, params, None

# -----------------------------------------------------------------------------
# QUEUE / WORKER
# -----------------------------------------------------------------------------
async def worker_loop(app: web.Application):
    pending = app["pending"]
    active = app["active"]
    lock = app["lock"]
    while True:
        job = None
        async with lock:
            if pending:
                job = pending.pop(0)
                job["status"] = "processing"
        if job is None:
            await asyncio.sleep(0.1)
            continue
        await run_job(job)
        if job["status"] == "ready":
            app["results"][job["uid"]] = {"path": job["file_path"], "ts": time.time()}
            asyncio.create_task(delayed_cleanup(app, job["uid"], READY_TIMEOUT))
        elif job["status"] == "fail":
            asyncio.create_task(delayed_cleanup(app, job["uid"], FAIL_TIMEOUT))

async def run_job(job: dict):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
    out_path = tmp.name
    tmp.close()
    try:
        ok, msg = await asyncio.to_thread(
            img2glb, image=job["image"], output_path=out_path, **job["params"]
        )
        if ok:
            job["file_path"] = out_path
            job["status"] = "ready"
        else:
            job["error"] = msg
            job["status"] = "fail"
            if os.path.exists(out_path):
                os.remove(out_path)
    except Exception as e:
        job["error"] = str(e)
        job["status"] = "fail"
        if os.path.exists(out_path):
            os.remove(out_path)

async def delayed_cleanup(app: web.Application, uid: str, delay: float):
    await asyncio.sleep(delay)
    entry = app["results"].pop(uid, None)
    if entry and os.path.exists(entry["path"]):
        os.remove(entry["path"])
    app["active"].pop(uid, None)

# -----------------------------------------------------------------------------
# SSE
# -----------------------------------------------------------------------------
def sse_payload(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")

async def stream_events(request: web.Request, job: dict):
    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })
    await resp.prepare(request)

    app = request.app
    pending = app["pending"]
    lock = app["lock"]

    try:
        while True:
            status = job["status"]
            if status == "pending":
                async with lock:
                    try:
                        pos = pending.index(job)
                    except ValueError:
                        pos = 0
                await resp.write(sse_payload({"status": "in_queue", "pos": pos}))
                await asyncio.sleep(1.0)
            elif status == "processing":
                await resp.write(sse_payload({"status": "processing"}))
                await asyncio.sleep(1.0)
            elif status == "ready":
                await resp.write(sse_payload({
                    "status": "ready",
                    "uid": job["uid"],
                    "timeout": READY_TIMEOUT,
                }))
                break
            elif status == "fail":
                await resp.write(sse_payload({
                    "status": "fail",
                    "error": job.get("error", "unknown error"),
                    "timeout": FAIL_TIMEOUT,
                }))
                break
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        pass
    return resp

# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------
async def handle_queue(request: web.Request):
    app = request.app
    lock = app["lock"]

    try:
        data = await request.json()
    except Exception as e:
        return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

    img, params, err = validate_payload(data)
    if err:
        return web.json_response({"error": err}, status=400)

    uid = uuid.uuid4().hex
    job = {
        "uid": uid,
        "status": "pending",
        "image": img,
        "params": params,
        "file_path": None,
        "error": None,
    }

    async with lock:
        processing = sum(1 for j in app["active"].values() if j["status"] == "processing")
        if len(app["pending"]) + processing >= MAX_QUEUE:
            return web.json_response({"error": "Queue is full"}, status=503)
        app["pending"].append(job)
        app["active"][uid] = job

    return await stream_events(request, job)

async def handle_download(request: web.Request):
    uid = request.match_info["uid"]
    entry = request.app["results"].get(uid)
    if not entry or not os.path.exists(entry["path"]):
        return web.Response(status=404, text="Not found")
    with open(entry["path"], "rb") as f:
        body = f.read()
    return web.Response(body=body, content_type="model/gltf-binary")

async def on_startup(app: web.Application):
    app["pending"] = []
    app["active"] = {}
    app["results"] = {}
    app["lock"] = asyncio.Lock()
    app["worker_task"] = asyncio.create_task(worker_loop(app))

app = web.Application()
app.on_startup.append(on_startup)
app.add_routes([
    web.post("/queue", handle_queue),
    web.get("/uid/{uid}", handle_download),
])

if __name__ == "__main__":
    web.run_app(app, host=HOST, port=PORT)
