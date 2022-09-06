import argparse
import time
import os
import requests
import json

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


def load_model():
    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True)
    pipe = pipe.to("cuda")

    return pipe

def run_model(pipe, prompt, save_image=False):
    with autocast("cuda"):
        image = pipe(prompt)["sample"][0]

    if save_image:
        ts = str(int(time.time()))
        image_name = f"sample-{prompt[:100].replace(' ', '_')}-{ts}.png"
        print(f"Image name is {image_name}")
        image.save(image_name)
    
    return image

def add_prompt_modifiers(plain_prompt):
    OPENAI_TOKEN = os.environ['OPENAI_TOKEN']

    with open('effective_prompts_fs.txt', 'r') as f:
        prefix = f.read()
    prompt = prefix + '\n' + plain_prompt

    response = requests.post(
        "https://api.openai.com/v1/completions",
        headers={
            'authorization': "Bearer " + OPENAI_TOKEN,
            "content-type": "application/json",
        },
        json={
            "model": "davinci",
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7,
            "stop": "\n",
        })

    text = response.text
    try:
        result = json.loads(text)
    except:
        raise Exception(f'Cannot load: {text}, {response}')

    prompt_modifiers = result['choices'][0]['text']
    engineered_prompt = plain_prompt + prompt_modifiers
    print(f'New engineered prompt: {engineered_prompt}')
    return engineered_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", type=str, help="Prompt into model")
    args = parser.parse_args()

    prompt = add_prompt_modifiers(args.prompt)
    pipe = load_model()
    run_model(pipe, prompt, save_image=True)


if __name__ == "__main__":
    main()