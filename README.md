## Long Stable Diffusion
### Long-form text inputs to images that accompany the text
e.g. story -> illustrations

Right now, Stable Diffusion can only take in a short prompt. What if you want to illustrate a full story? Cue Long Stable Diffusion, a pipeline of generative models to do just that with just a bash script!

**Steps**
1. Start with long-form text that you want accompanying images for, e.g. a story to illustrate. Put a `.txt` file in the `texts` folder.
2. Ask GPT-3 for several illustration ideas for beginning, middle, end, via the OpenAI API. This is important, because otherwise GPT-3 will generate mostly for the beginning only. These are written to the folder `image_prompts`.
3. "Translate" the ideas from English to "prompt-English", e.g. add suffixes like `trending on art station` for better results. I say translate in quotes, because it's English to English, but prompt English can sound funky. It's for a good cause which is to nudge the model into a good place and make the resulting images look better without sacrificing semantics.
4. Put the prompt-English prompts into Stable Diffusion to generate the images. Images are saved to the `images` folder, wiht names based on prompt and timestamp.
5. I also wanted a `.docx` file with the images and their prompts, for easy copy-pasting. These are outputed to the `docx` folder.

Note: Multi-processing is optimized for 2 Titan RTXs, with 24GB RAM each. Changing the number of GPUs to parallelize on is a simple edit in `run_longsd.sh`: just copy the first line and change CUDA_VISIBLE_DEVICES to the appropriate GPU id. Changing the number of processes for each GPU is an argument that can be passed in through `run_longsd.sh` as `-n <num_processes_per_gpu>` for each run. This is an int used in `longsd.py`. I've found that my GPUs can handle 3, but are happier with 2.

**Purpose**
I made this to automate my self, ie. prompt AI for illustrations to accompany AI-generated stories, for the [AI Stories Podcast](). Come check us out! And please suggest ways to improve -- comments and pull requests are always welcome :) 

This was also just a weekend hackathon project to reward myself for doing a lot of work the past couple of months, and for feeling guilty about not using my wonderful and beautiful Titan RTXs to their full potential.

## Run
This bash script runs what you need. It assumes 2 GPUs with 24GB memory each. See the note above, under Steps, to change this assumption for your compute needs. I had too much fun with multiprocessing and making it faster.
`bash run_longsd.sh <name_of_txtfile_in_texts_dir>`

**What you need**
OK, before you run it like that. 
- Install the requirements
- Make sure you set your OpenAI API key, e.g. in terminal `export OPENAI_TOKEN=<your_token>`
- Then, put your favorite story or article in a text file in `texts`

### Files and folders
`run_longsd.sh`: This is the main entry script into the program to parallelize across GPUs easily.
`longsd.py`: Where most of the magic happens: getting image prompts from GPT-3, making images from those prompts (using stable diffusion, multithreading), saving all those and also dumping those images and prompts to a docx file. This is what `run_longsd.sh` calls.
`sd.py`: Just runs stable diffusion if you want to use it by itself (I do). `longsd.py` calls it.
`dump_docx.py`: Just dumps image prompts and images into a single docx for a particular text. Again, it's useful if you want to use it by itself on the saved images and prompts. I do, because I'm actually overwriting the file when multiprocessing and sometimes will just use this as a postprocessing step. Yes, you can join those and change that but I don't really care, since sometimes my GPUs misbehave and I'll need to rerun it anyways.

`texts/`: Folder to put your texts in, as a `.txt` file.
`image_prompts/`: Generated image prompts by GPT-3 based on your text.
`images`: Generated images by Stable Diffusion based on GPT-3's image prompts.
`docx/`: Microsoft Word document for a text with images and their prompts all in one.


### TODOs
- [x] Pipeline of asking GPT3 for image prompts
- [x] Image prompts to stable diffusion
- [x] Multiprocessing to max out a single GPU
- [x] GPU multiprocessing stable diffusion
- [x] Docx dump of images and image prompts
- [ ] Translation layer between English prompt and "prompt English" (lexica)
- [ ] Walkthrough video of code
- [x] Flesh out readme
- [ ] Open source