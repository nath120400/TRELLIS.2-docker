import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import base64
import io
import asyncio
from aiohttp import web
from PIL import Image
import tempfile

import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# Load model
pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
pipeline.cuda()

def img2glb(
    # input image
    image: Image.Image,
    preprocess_image: bool = True,
    # model parameters
    seed: int = 42,
    resolution: str = "1024", # '512', '1024', '1024_cascade', '1536_cascade'
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
    # output parameters
    decimation_target: int = 1_000_000,
    texture_size: int = 4096,
    output_path: str = None
):
    # Check resolution string
    if resolution not in ['512', '1024', '1024_cascade', '1536_cascade']:
        return 0, "WRONG RESOLUTION : ACCEPTED VALUES['512', '1024', '1024_cascade', '1536_cascade']"

    try:
        image = image.convert("RGBA")
        
        # Run pipeline
        outputs, latents = pipeline.run(
            image,
            seed=seed,
            preprocess_image=True,
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
        return 0, f"FAILED PIPELINE : {e}"
    
    try:
        # Retrieve mesh
        mesh = outputs[0]
        mesh.simplify(16777216)
        _, _, res = latents 
        
        # Clean vram
        torch.cuda.empty_cache()

        # GLB extraction
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
        return 0, f"FAILED EXTRACTION : {e}"
    
    # Export
    try:
        glb.export(output_path, extension_webp=True)

        # Clean vram
        torch.cuda.empty_cache()
        return 1, "SUCCESS"
    except Exception as e:
        return 0, f"FAILED EXPORT : {e}"

# Our little global memory
state = {
    "status": "IDLE", # IDLE, PROCESSING, DONE, ERROR
    "file": None,
    "error": ""
}

async def handle_post(request):
    if state["status"] == "PROCESSING":
        return web.Response(status=102, text="Generation already in progress")

    try:
        data = await request.json()
        image_b64 = data.pop("image")
        
        # Decode image
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return web.Response(status=400, text=f"Bad request: {e}")

    # Clean up the old file if it's still hanging around
    if state["file"] and os.path.exists(state["file"]):
        os.remove(state["file"])

    # Temp file for export
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
    state["file"] = temp.name
    state["status"] = "PROCESSING"
    state["error"] = ""

    # Launch in background (without blocking the web server)
    asyncio.create_task(run_generation(img, data, state["file"]))

    return web.Response(status=200, text="Generation started!")

async def run_generation(img, kwargs, output_path):
    try:
        # Using to_thread to avoid blocking the async loop
        success, msg = await asyncio.to_thread(
            img2glb, image=img, output_path=output_path, **kwargs
        )
        if success:
            state["status"] = "DONE"
        else:
            state["status"] = "ERROR"
            state["error"] = msg
    except Exception as e:
        state["status"] = "ERROR"
        state["error"] = str(e)

async def handle_get(request):
    if state["status"] == "IDLE":
        return web.Response(status=404, text="Nothing to retrieve")
        
    elif state["status"] == "PROCESSING":
        return web.Response(status=102, text="Processing...")
        
    elif state["status"] == "ERROR":
        msg = state["error"]
        state["status"] = "IDLE"
        return web.Response(status=500, text=f"Crash: {msg}")
        
    elif state["status"] == "DONE":
        if state["file"] and os.path.exists(state["file"]):
            # Read the file so we can delete it immediately
            with open(state["file"], "rb") as f:
                content = f.read()
            
            os.remove(state["file"])
            state["status"] = "IDLE"
            state["file"] = None
            
            return web.Response(body=content, content_type="model/gltf-binary")
        else:
            state["status"] = "IDLE"
            return web.Response(status=404, text="File not found")

app = web.Application()
app.add_routes([
    web.post('/', handle_post),
    web.get('/', handle_get)
])

if __name__ == '__main__':
    web.run_app(app, port=7861)