# Llama3.1-70B Inference
code in this directory was modified from
- https://github.com/meta-llama/llama-models
- https://github.com/meta-llama/llama-stack

### prerequisites

- **miniconda**(makes dependency management easier)\
https://docs.anaconda.com/miniconda/#quick-command-line-install
- **pytorch**\
https://pytorch.org/get-started/locally/
- **llama-models** (tokenizer)\
https://github.com/meta-llama/llama-models

### implementations
- v0_cpu_only: the whole model on CPU

### weights preprocessing
1. follow instructions here to download the model weights:
https://github.com/meta-llama/llama-models/blob/main/README.md

2. run weights preprocessing script (original weights are designed to be ran on 8 GPUs)
```sh
# command below should be ran inside llama3_1/
# remember to change input and output paths accordingly
python weights_preprocessor.py --model-path ~/Meta-Llama3.1-70B/
```

### sample command
```sh
# command below should be ran inside llama3_1/
# optional: add --hide-resp flag to hide input / output texts during performance analysis
python ./v0_cpu_only/model.py --model-path ~/Meta-Llama3.1-70B/ --prompt "to be, or not to be" --max-gen-len 32
```

### known issues
refer to comments in Llama3_1.generate() function for more details
- when batch size > 1
    - prompts with length < max will generate more tokens than max_gen_len
    - when prompts within the same batch have unequal lengths, the longer ones' excess tokens will be processed one-by-one along with the shortest prompt's token generation
