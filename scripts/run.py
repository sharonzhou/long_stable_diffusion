import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/scripts')
import argparse
import json
import time
import requests
import logging
import string
import glob
from nltk.tokenize import sent_tokenize, word_tokenize

import torch
import torch.multiprocessing as torch_mp

from stable_diffusion import load_model as load_sd, run_model as run_sd, add_prompt_modifiers
from gen_docx import dump_images_captions_docx

OPENAI_TOKEN = os.environ['OPENAI_TOKEN']
SECTIONS = ["start", "middle", "end"]
EXTRACT_LENGTH = 100
logger = logging.getLogger('run_sd')
logging.basicConfig(level=logging.DEBUG)
with torch.no_grad():
    torch.cuda.empty_cache()
torch_mp.set_start_method("spawn", force=True)
SD_MODEL = None

def query_gpt3(prompt):
    response = requests.post(
        "https://api.openai.com/v1/completions",
        headers={
            'authorization': "Bearer " + OPENAI_TOKEN,
            "content-type": "application/json",
        },
        json={
            "model": "text-davinci-002",
            "prompt": prompt,
            "max_tokens": 150,
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
    return result

def generate_image_prompts_with_sections(text):
    suffix_template = "\n\nRecommend five different detailed, logo-free, sign-free images to accompany the previous text that illustrate the {} of this text: 1)"
    image_prompts = { s: [] for s in SECTIONS }
    for section in SECTIONS:
        suffix = suffix_template.format(section)
        prompt = text + suffix
        result = query_gpt3(prompt)
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
    return image_prompts

def generate_image_prompts_with_extracts(text):
    image_prompts = []
    with open('inputs/extracts_prompt_prefix.txt','r') as f:
        promt_prefix = f.read()
    extracts = []
    text_sentences = sent_tokenize(text)
    current_extract = None
    current_extract_num_words = 0
    for sentence in text_sentences:
        if current_extract is None:
            current_extract = sentence
        else:
            current_extract+=' '+sentence
        current_extract_num_words+=len(word_tokenize(sentence))
        if current_extract_num_words > EXTRACT_LENGTH:
            extracts.append(current_extract)
            current_extract = None
            current_extract_num_words = 0

    if current_extract is not None:
        extracts.append(current_extract)

    for extract in extracts:
        prompt = promt_prefix+'\n~\n'+extract+'---'
        result = query_gpt3(prompt)
        image_prompts.append(result.strip())
    return {'EXTRACTS': image_prompts}

def generate_image_prompts(method, text, filename):
    if method == "sections":
        image_prompts = generate_image_prompts_with_sections(text)
    if method == "extracts":
        image_prompts = generate_image_prompts_with_extracts(text)

    # Store image prompts
    filepath = f'outputs/image_prompts/{filename}.json'
    logger.debug(f'Writing image prompts to {filepath}...')
    with open(filepath, 'w') as f:
        f.write(json.dumps(image_prompts, indent=4))

    filepath_all = f'outputs/image_prompts/{filename}-all.txt'
    logger.debug(f'Writing image prompts to {filepath_all}...')
    with open(filepath_all, 'a') as f:
        f.write(json.dumps(image_prompts, indent=4))
        f.write('\n')
        f.write(json.dumps(image_prompts, indent=4))
        f.write('\n')

    logger.debug(image_prompts)

    return image_prompts

def make_image_prompts(method, filename, text, overwrite_prompts):
    filename = filename.split('.')[0]

    engineered_filepath = f"outputs/engineered_image_prompts/{filename}_{method}.json"
    engineered_filepath_all = f"outputs/engineered_image_prompts/{filename}_{method}-all.txt"

    if os.path.exists(engineered_filepath) and not overwrite_prompts:
        logger.debug(f'Reading from existing {engineered_filepath}...')
        with open(engineered_filepath) as f:
            engineered_prompts = json.load(f)
    else:
        image_prompts = generate_image_prompts(method, text, filename)

        # TODO: extractive summarization from long-form text as additional prompts to engineer and input into Stable Diffusion

        # Engineer prompts (add modifiers to image prompts)
        if method == 'sections':
            engineered_prompts = { s: [] for s in SECTIONS }
            for section, prompts in image_prompts.items():
                for prompt in prompts:
                    engineered_prompt = add_prompt_modifiers(prompt)
                    engineered_prompts[section].append(engineered_prompt)
        else:
            engineered_prompts = image_prompts

        if not os.path.isfile(engineered_filepath_all,):
            with open(engineered_filepath_all, 'w') as fp:
                pass

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

def _run_text_to_image(args):
    prompt = args['prompt']
    section = args['section']
    save_folder = args['save_folder']

    image = run_sd(SD_MODEL, prompt) # PIL output

    save_prompt_name = prompt[:100].replace(' ', '_')
    image_name = f'{section}-{save_prompt_name}-{str(int(time.time()))}'
    image_path = f'{save_folder}/{image_name}.png'
    image.save(image_path)

    return (prompt, image_path)

def generate_images(sd_inputs, num_processes):
    global SD_MODEL 
    SD_MODEL = load_sd()
    pool = torch_mp.Pool(processes=num_processes)
    prompts_and_image_paths = pool.map(_run_text_to_image, sd_inputs)
    pool.close()
    pool.join()
    return prompts_and_image_paths

def setup(file):
    save_folder = 'outputs/images/' + file.split('.')[0].replace(' ', '-')
    os.makedirs(save_folder, exist_ok=True)
    logger.debug(f'Using folder to save: {save_folder}')

    filepath = f'inputs/{file}' if '.' in file else f'inputs/{file}.txt'
    with open(filepath, 'r') as f:
        text = f.read()

    return text, save_folder

def prepare_sd_inputs(method, image_prompts, save_folder):
    sd_inputs = []

    if method == 'sections':
        section_counts = {}
        for s in SECTIONS:
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
    elif method == 'extracts':
        for prompt in image_prompts.values():
            sd_input = {
                'prompt': prompt,
                'section': 'EXTRACTS',
                'save_folder': save_folder,
            }
    logger.debug(sd_inputs)
    return sd_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        "-f",
        default=["sample_input.txt"],
        type=str,
        nargs='+',
        help="File for text"
    )
    parser.add_argument(
        "--overwrite_prompts",
        "-op",
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
        default=200,#TODO not using right now
        type=int,
        help="If using extract method, how many words to include in each extract.")
    parser.add_argument(
        "--output",
        "-o",                                 #TODO
        choices=["txt", "images", "html", "docx", "markdown", "latex", "pdf"],
        default="images",
        help="Where to put resulting images (see README)")
    args = parser.parse_args()
    print(args.method)
    files = args.files
    for file in files:
        text, save_folder = setup(file)
        image_prompts = make_image_prompts(args.method, file, text, overwrite_prompts=args.overwrite_prompts)
        sd_inputs = prepare_sd_inputs(args.method, image_prompts, save_folder)
        prompts_and_image_paths = generate_images(sd_inputs, args.num_gpu_processes)
        dump_images_captions_docx(file, prompts_and_image_paths)

    logger.info('All complete')


if __name__ == "__main__":
    main()
