import os
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

def download_model():
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    model_id = "stabilityai/stable-diffusion-2-1"
    model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=HF_AUTH_TOKEN)
    model.scheduler = EulerDiscreteScheduler.from_config(model.scheduler.config)


if __name__ == "__main__":
    download_model()
