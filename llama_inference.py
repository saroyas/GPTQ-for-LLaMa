# ... (imports and other functions)

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
