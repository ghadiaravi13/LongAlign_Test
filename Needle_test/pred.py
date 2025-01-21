import logging
import yaml
import os
import glob
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.generation.utils import GenerationConfig
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn()

os.environ['HF_HOME'] = "/work/10198/ghadiaravi13/ls6/HopFormer/HF_Llama3/HF_cache"
cache_dir = "/work/10198/ghadiaravi13/ls6/HopFormer/HF_Llama3/HF_cache/"

def pred(model_name, model, tokenizer, input_data, device, max_new_tokens=1024, temperature=0.1):
    prompt = input_data[0]['content']+'\n'+input_data[1]['content']
    history = []
    if "internlm" in model_name or "chatglm" in model_name or "longalign-6b" in model_name:
        response, history = model.chat(tokenizer, prompt, history=history, max_new_tokens=max_new_tokens, temperature=temperature)
        return response
    elif "longalign-7b" in model_name or "longalign-13b" in model_name:
        if history == []:
            prompt = f"[INST]{prompt}[/INST]"
        else:
            prompt = history+"\n\n"+f"[INST]{prompt}[/INST]"
    elif "mistral" in model_name or "mixtral" in model_name:
        if history == []:
            prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            prompt = history+f"</s> [INST] {prompt} [/INST]"
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    context_length = input.input_ids.shape[-1]
    output = model.generate(
        **input,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        temperature=temperature,
    )[0]
    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    return pred.strip()

def load_model_and_tokenizer(path, config, device):
    valid_path = path.lower()
    if "longchat" in valid_path or "vicuna" in valid_path:
        from fastchat.model import load_model
        model, _ = load_model(path, cache_dir = cache_dir, device='cpu', num_gpus=0, load_8bit=False, cpu_offloading=False, debug=False)
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    elif "mistral" in valid_path or "mixtral" in valid_path:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, cache_dir = cache_dir, config=config, use_flash_attention_2=True, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device) #, device_map="auto")
        model.generation_config = GenerationConfig.from_pretrained(path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, cache_dir = cache_dir, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)# , device_map="auto")
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    with open('config-pred.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model_provider = config['model']['model_provider']
    model_name = config['model']['model_name']
    model_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model_config._attn_implementation = "flash_attention_2"
    model_config.hopformer = config['model']['hopformer_config']
    model_config.snapkv = config['model']['snapkv']
    if model_config.snapkv:
        model_config.window_size = config['model']['snapkv']['window_size']
        model_config.max_capacity_prompt = config['model']['snapkv']['sim_threshold']
        model_config.kernel_size = 5
        model_config.pooling = "avgpool"
    prompt_dir = config['prompt_dir']
    save_dir = config['save_dir']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if config['model']['hopformer_config']:
        logfile = f"{save_dir}/{model_name.split('/')[1]}_ws{config['model']['hopformer_config']['window_size']}_st{config['model']['hopformer_config']['sim_threshold']}.log"
    else:
        logfile = f"{save_dir}/{model_name.split('/')[1]}_no_hopf.log"
    logging.basicConfig(filename=logfile,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        force=True)
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = load_model_and_tokenizer(model_name, model_config, device)

    for filename in glob.glob(f'{prompt_dir}/{model_provider}_*_prompts.json'):
        with open(filename, 'r') as f:
            prompts = json.load(f)

        result = pred(model_name.lower(), model, tokenizer, prompts, device)

        basename = os.path.basename(filename)
        newname = basename.replace('.json', '.txt').replace('_prompts', '')
        with open(f'{save_dir}/{newname}', 'w') as f:
            f.write(result)


