# Mixtral8x7B Inference
code in this directory was modified from https://github.com/mistralai/mistral-inference

### prerequisites

- **miniconda**(makes dependency management easier)\
https://docs.anaconda.com/miniconda/#quick-command-line-install
- **pytorch**\
https://pytorch.org/get-started/locally/
- **mistral-common** (tokenizer)\
https://github.com/mistralai/mistral-common
- **xformers** (fast transformer related functions by Meta AI)\
https://github.com/facebookresearch/xformers

### implementations
- v0_cpu_experts: weights and computation on CPU
- v1_gpu_experts: experts initially stored on host memory, memcpy to GPU for computation
- v2_static_collab: 1 expert on GPU, 7 on CPU. Computation happen where the weights are
- v3_dnm_collab: same as v2, but issues memcpy when communication overhead can be covered by computation savings

### weights preprocessing
1. follow instructions here to download the model weights:
https://github.com/mistralai/mistral-inference

2. run weights preprocessing script
```sh
# command below should be ran inside mixtral/
# remember to change input and output paths accordingly
python weights_preprocessor.py --input-path ~/Mixtral-8x7B-Instruct-v0.1-Official/ --output-path ~/Mixtral-8x7B-Instruct-v0.1-Official/
```

### sample command
```sh
# command below should be ran inside mixtral/
# optional: add --hide-resp flag to hide input / output texts during performance analysis
python ./v0_poc/solo_gpu_model.py --model-path ~/Mixtral-8x7B-Instruct-v0.1-Official/ --prompt-path ../prompts/diverse_short.json --n-prompts 64 --max-tokens 128
```
