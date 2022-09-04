import argparse
import time

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


def run_model_from_prompt(prompt, save_image=False):
    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True)

    pipe = pipe.to("cuda")

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

    run_model_from_prompt(args.prompt)


if __name__ == "__main__":
    main()