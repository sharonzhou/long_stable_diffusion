import pandas as pd
import glob
import argparse


def clean_prompts(csv_dir):
    cleaned_prompts = []

    for filepath in glob.glob(f'{csv_dir}/*.csv'):
        df = pd.read_csv(filepath)

        for _, row in enumerate(df.iterrows()):
            prompt = str(row[1]['Content'])
            if '[-h]' in prompt or (prompt[0] == '-' and prompt[2] == ' ') or (prompt[:2] == '--' and prompt[3] == ' '):
                continue
            # Split on !dream command
            elif '!dream' in prompt:
                cleaned_prompt = prompt.split('!dream')[-1].strip()
            elif '! dream' in prompt:
                cleaned_prompt = prompt.split('! dream')[-1].strip()

            # Remove accidental pasting of just params
            if not cleaned_prompt or cleaned_prompt == '-h' or cleaned_prompt == '--help':
                continue

            # Strip quotes
            if cleaned_prompt[0] == '"':
                cleaned_prompt = cleaned_prompt.split('"')[1].strip()
            elif cleaned_prompt[0] == '“':
                cleaned_prompt = cleaned_prompt.split('”')[0].replace('“', '').strip()

            # Remove more params
            cleaned_prompt = cleaned_prompt.split(' -')[0]

            print(cleaned_prompt)
            cleaned_prompts.append(cleaned_prompt)
    
    # Remove duplicates
    cleaned_prompts = list(set(cleaned_prompts))
    return cleaned_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lexica_dir_path", 
        "-p", 
        type=str, 
        required=True, 
        help="Path to Lexica folder with csvs of dump"
    )
    args = parser.parse_args()

    csv_dir = args.lexica_dir_path
    cleaned_prompts = clean_prompts(csv_dir)

    # Write to file
    with open('lexica_prompts.txt', 'w') as f:
        for p in cleaned_prompts:
            f.write(f'{p}\n') # will mess up multiline prompts


if __name__ == "__main__":
    main()