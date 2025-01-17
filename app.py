import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import base64
from io import BytesIO
import os

def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    model_id = "stabilityai/stable-diffusion-2-1"

    model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=HF_AUTH_TOKEN)
    model.scheduler = EulerDiscreteScheduler.from_config(model.scheduler.config)
    model = model.to("cuda")
    
def inference(model_inputs:dict) -> dict:
    global model

    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 768)
    width = model_inputs.get('width', 768)
    steps = model_inputs.get('steos', 5)
    guidance_scale = model_inputs.get('guidance_scale', 9)
    seed = model_inputs.get('seed', None)

    if not prompt: return {'message': 'No prompt was provided'}
    
    generator = None
    if seed: generator = torch.Generator("cuda").manual_seed(seed)
    
    with autocast("cuda"):
        image = model(prompt, guidance_scale=guidance_scale, height=height, width=width, num_inference_steps=steps, generator=generator).images[0]
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
