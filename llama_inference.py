import argparse

import torch
import torch.nn as nn
import quant

from gptq import GPTQ
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders
import transformers
from transformers import AutoTokenizer


def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model

def main(model, wbits=16, groupsize=-1, load='', text='', min_length=10, max_length=50,
         top_p=0.95, temperature=0.8, device=-1, fused_mlp=True):
    
    if type(load) is not str:
        load = load.as_posix()

    if load:
        model = load_quant(model, load, wbits, groupsize, fused_mlp=fused_mlp)
    else:
        model = get_llama(model)
        model.eval()

    model.to(DEV)
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(DEV)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=min_length,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
        )
    print(tokenizer.decode([el.item() for el in generated_ids[0]]))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # ... (same argparse code to define parser)

    args = parser.parse_args()

    main(args.model, wbits=args.wbits, groupsize=args.groupsize, load=args.load,
         text=args.text, min_length=args.min_length, max_length=args.max_length,
         top_p=args.top_p, temperature=args.temperature, device=args.device, fused_mlp=args.fused_mlp)
