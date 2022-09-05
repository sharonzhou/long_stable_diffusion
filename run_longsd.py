import argparse
import json
import time
import requests
import os
import logging
import string

import torch
import torch.multiprocessing as mp

from sd import load_model as load_sd, run_model as run_sd


OPENAI_TOKEN = os.environ['OPENAI_TOKEN']
logger = logging.getLogger('run_longsd')
logging.basicConfig(level=logging.DEBUG)
sd_model = load_sd()

def get_image_prompts(filename, text, overwrite_prompts):
    filepath = f"image_prompts/{filename.split('.')[0]}.json"
    filepath_all = f"image_prompts/{filename.split('.')[0]}-all.txt"
    if os.path.exists(filepath) and not overwrite_prompts:
        logger.debug(f'Reading from existing {filepath}...')
        with open(filepath) as f:
            image_prompts = json.load(f)
    else:
        suffix_template = "\n\nRecommend five different detailed, logo-free, sign-free images to accompany the previous text that illustrate the {} of this text: 1)"
        sections = ["start", "middle", "end"]
        image_prompts = { s: [] for s in sections }
        for section in sections:
            suffix = suffix_template.format(section)
            prompt = text + suffix

            logger.debug(f'Generating image prompts for {section}...')

            response = requests.post("https://api.openai.com/v1/completions",
                                    headers={
                                        'accept': "*/*",
                                        "accept-language": "en-US,en;q=0.9",
                                        'authorization': "Bearer " + OPENAI_TOKEN,
                                        "content-type": "application/json",
                                        "sec-fetch-dest": "empty",
                                        "sec-fetch-mode": "cors",
                                        "sec-fetch-site": "same-origin",
                                        "sec-gpc": "1",
                                    },
                                    json={
                                        "model": "text-davinci-002",
                                        "prompt": prompt,
                                        "max_tokens": 256,
                                        "temperature": 0.8,
                                    })
            raw_response = response.text
            logger.debug(raw_response)
            try:
                result = json.loads(raw_response)
            except:
                logger.debug('Cannot load parse raw response on get ideas', raw_response)
                return 'Error', 500
            
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
        logger.debug(f'Writing image prompts to {filepath_all}...')
        with open(filepath_all, 'a') as f:
            f.write(json.dumps(image_prompts, indent=4))
        logger.debug(f'Writing image prompts to {filepath}...')
        with open(filepath, 'w') as f:
            f.write(json.dumps(image_prompts, indent=4))
    
    logger.debug(image_prompts)
    return image_prompts


def run_text_to_image(args, save_images=True):
    prompt = args['prompt']
    section = args['section']
    save_folder = args['save_folder']

    image = run_sd(sd_model, prompt) # PIL
    
    if save_images:
        save_prompt_name = prompt[:100].replace(' ', '_')
        image_name = f'{section}-{save_prompt_name}-{str(int(time.time()))}'
        image_path = f'{save_folder}/{image_name}.png' 
        image.save(image_path)

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", "-f", type=str, required=True, nargs='+', help="File for text")
    parser.add_argument("--overwrite_prompts", "-o", action='store_true', help="Overwrite json file image prompts")
    args = parser.parse_args()

    torch.cuda.empty_cache()    
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(processes=3)

    files = args.files
    for file in files:

        save_folder = 'images/' + file.split('.')[0].replace(' ', '-')
        os.makedirs(save_folder, exist_ok=True)
        logger.debug(f'Using folder to save: {save_folder}')

        filepath = f'texts/{file}' if '.' in file else f'texts/{file}.txt'
        with open(filepath, 'r') as f:
            text = f.read()

        image_prompts = get_image_prompts(file, text, overwrite_prompts=args.overwrite_prompts)
    
        sd_inputs = []
        for section, prompts in image_prompts.items():
            for prompt in prompts:
                sd_input = {
                    'prompt': prompt,
                    'section': section,
                    'save_folder': save_folder,
                }
                
                sd_inputs.append(sd_input)
        logger.debug(sd_inputs)

        images = pool.map(run_text_to_image, sd_inputs)
        pool.close()
        pool.join()
        logger.info('Complete')
        

if __name__ == "__main__":
    main()