import argparse
import json
import time
import requests
import os
import logging
import string
import glob

import torch
import torch.multiprocessing as torch_mp

from scripts.stable_diffusion import load_model as load_sd, run_model as run_sd, add_prompt_modifiers
from scripts.dump_docx import dump_images_captions_docx

OPENAI_TOKEN = os.environ['OPENAI_TOKEN']
SECTIONS = ["start", "middle", "end"]
logger = logging.getLogger('run_sd')
logging.basicConfig(level=logging.DEBUG)

def generate_image_prompts(method, text, filename):
    with torch.no_grad():
        torch.cuda.empty_cache()
    torch_mp.set_start_method("spawn", force=True)

    sd_model = load_sd()
    if method == "summary":
        suffix_template = "\n\nRecommend five different detailed, logo-free, sign-free images to accompany the previous text that illustrate the {} of this text: 1)"
    if method == "extracts":
        suffix_template = "Recommend a description of an image to illustrate for the following paragraphs:"

    image_prompts = { s: [] for s in sections }
    for section in SECTIONS:
        suffix = suffix_template.format(section)
        prompt = text + suffix

        logger.debug(f'Generating image prompts for {section}...')

        response = requests.post(
            "https://api.openai.com/v1/completions",
            headers={
                'authorization': "Bearer " + OPENAI_TOKEN,
                "content-type": "application/json",
            },
            json={
                "model": "text-davinci-002",
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0.8,
            }
        )
        text = response.text
        logger.debug(text)
        try:
            result = json.loads(text)
        except:
            raise Exception(f'Cannot load: {text}, {response}')

        result = result['choices'][0]['text']
        result_list = result.strip().split(")") # removes space and number

        clean_result_list = []
        for i, r in enumerate(result_list):
            res = r.strip()
            if not res:
                continue

            if i < len(result_list) - 1:
                res = res[:-2]

            # Remove punctuation
            res = res.translate(str.maketrans('', '', string.punctuation))

            clean_result_list.append(res)
        image_prompts[section].extend(clean_result_list)

    # Store image prompts
    filepath = f'image_prompts/{filename}.json'
    logger.debug(f'Writing image prompts to {filepath}...')
    with open(filepath, 'w') as f:
        f.write(json.dumps(image_prompts, indent=4))

    filepath_all = f'image_prompts/{filename}-all.txt'
    logger.debug(f'Writing image prompts to {filepath_all}...')
    with open(filepath_all, 'a') as f:
        f.write(json.dumps(image_prompts, indent=4))
        f.write('\n')
        f.write(json.dumps(image_prompts, indent=4))
        f.write('\n')

    logger.debug(image_prompts)

    return image_prompts

def make_image_prompts(filename, text, overwrite_prompts):
    filename = filename.split('.')[0]

    engineered_filepath = f"engineered_image_prompts/{filename}.json"
    engineered_filepath_all = f"engineered_image_prompts/{filename}-all.txt"

    if os.path.exists(engineered_filepath) and not overwrite_prompts:
        logger.debug(f'Reading from existing {engineered_filepath}...')
        with open(engineered_filepath) as f:
            engineered_prompts = json.load(f)
    else:
        image_prompts = generate_image_prompts(text, filename)

        # TODO: extractive summarization from long-form text as additional prompts to engineer and input into Stable Diffusion

        # Engineer prompts (add modifiers to image prompts)
        engineered_prompts = { s: [] for s in sections }
        for section, prompts in image_prompts.items():
            for prompt in prompts:
                engineered_prompt = add_prompt_modifiers(prompt)
                engineered_prompts[section].append(engineered_prompt)

        # Store engineered image prompts
        logger.debug(f'Writing engineered image prompts to {engineered_filepath_all}...')
        with open(engineered_filepath_all, 'a') as f:
            f.write(json.dumps(engineered_prompts, indent=4))
            f.write('\n')
            f.write(json.dumps(engineered_prompts, indent=4))
            f.write('\n')

        logger.debug(f'Writing engineered image prompts to {engineered_filepath}...')
        with open(engineered_filepath, 'w') as f:
            f.write(json.dumps(engineered_prompts, indent=4))

    logger.debug(engineered_prompts)
    return engineered_prompts

def run_text_to_image(args):
    prompt = args['prompt']
    section = args['section']
    save_folder = args['save_folder']

    image = run_sd(sd_model, prompt) # PIL output

    save_prompt_name = prompt[:100].replace(' ', '_')
    image_name = f'{section}-{save_prompt_name}-{str(int(time.time()))}'
    image_path = f'{save_folder}/{image_name}.png'
    image.save(image_path)

    return (prompt, image_path)

def gpu_multiprocess(sd_inputs, num_processes):
    pool = torch_mp.Pool(processes=num_processes)
    prompts_and_image_paths = pool.map(run_text_to_image, sd_inputs)
    pool.close()
    pool.join()
    return prompts_and_image_paths

def setup(file):
    save_folder = 'images/' + file.split('.')[0].replace(' ', '-')
    os.makedirs(save_folder, exist_ok=True)
    logger.debug(f'Using folder to save: {save_folder}')

    filepath = f'texts/{file}' if '.' in file else f'texts/{file}.txt'
    with open(filepath, 'r') as f:
        text = f.read()

    return text, save_folder

def prepare_sd_inputs(image_prompts, save_folder):
    sd_inputs = []

    section_counts = {}
    for s in sections:
        section_counts[s] = len(glob.glob(f'{save_folder}/{s}-*.png'))

    # Generate, sorted by the section that has the least images generated
    for section in sorted(section_counts, key=lambda k: section_counts[k]):
        prompts = image_prompts[section]
        for prompt in prompts:
            sd_input = {
                'prompt': prompt,
                'section': section,
                'save_folder': save_folder,
            }

            sd_inputs.append(sd_input)
    logger.debug(sd_inputs)
    return sd_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        "-f",
        type=str,
        required=True,
        nargs='+',
        help="File for text"
    )
    parser.add_argument(
        "--overwrite_prompts",
        "-o",
        action='store_true',
        help="Overwrite json file image prompts"
    )
    parser.add_argument(
        "--num_gpu_processes",
        "-n",
        default=3,
        type=int,
        help="Num processes for gpu multiprocessing"
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["sections", "summary", "extracts", "summary+extracts"],
        default="sections",
        help="How to generate image prompts (see README)")
    parser.add_argument(
        "--extract_length",
        "-el",
        default=200,
        type=int,
        help="If using extract method, how many words to include in each extract.")
    parser.add_argument(
        "--output",
        "-o",                                 #TODO
        choices=["txt", "images", "html", "docx", "markdown", "latex", "pdf"],
        default="images",
        help="Where to put resulting images (see README)")
    args = parser.parse_args()

    files = args.files
    for file in files:
        text, save_folder = setup(file)
        image_prompts = make_image_prompts(file, text, overwrite_prompts=args.overwrite_prompts)
        sd_inputs = prepare_sd_inputs(image_prompts, save_folder)
        prompts_and_image_paths = gpu_multiprocess(sd_inputs, args.num_gpu_processes)
        dump_images_captions_docx(file, prompts_and_image_paths)

    logger.info('All complete')


if __name__ == "__main__":
    main()
