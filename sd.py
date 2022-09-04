import argparse
import time

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

def load_model():
    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        revision="fp16", 
        torch_dtype=torch.float16, 
        use_auth_token=True
    )
    pipe = pipe.to("cuda")

    return pipe

def run_model(pipe, prompt, save_image=False):
    with autocast("cuda"):
        image = pipe(prompt)["sample"][0]

    if save_image:
        image_id = str(int(time.time()))
        image_name = f"sample-{prompt[:100]}-{image_id}"
        print(f"Image name is {image_name}")
        image.save(image_name)
    
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="Prompt into model")
    args = parser.parse_args()

    pipe = load_model()
    run_model(pipe, args.prompt)


if __name__ == "__main__":
    main()