# from transformers import AutoModel
# 
# model_path = "Meta-Llama-3.1-70B-Instruct"
# model = AutoModel.from_pretrained(model_path)
# 
# pipeline = transformers.pipeline(
#     "text-generation", model=model, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
# )
# 
# pipeline("Hey how are you doing today?")

import torch
from torch import Tensor
from transformers import pipeline, AutoModel, AutoTokenizer, set_seed, LlamaForCausalLM, LlamaTokenizer, GenerationConfig

set_seed(123)

def run_pipeline(texts, run_original=True):
    # model_id = "Llama-3.2-1B"
    if run_original:
        model_id = "Llama-2-7b-hf"
    else:
        model_id = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_to_transformers/Llama-2-7b-hf"
    
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        # torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    for text in texts:
        print(pipe(text))


def run_model(texts):
    model_id = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_to_transformers/Llama-2-7b-hf"
    model_id = "Llama-2-7b-hf"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(model_id)
    model.to(device)
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    generation_config = GenerationConfig(
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        do_sample=True,
        temperature=0.6,
        max_length=4096,
        top_p=0.9
    )
    for text in texts:
        import pdb; pdb.set_trace()
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, generation_config=generation_config)
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        import pdb; pdb.set_trace()
        print(output_texts)

def run_fuji_model(texts):
    from axlearn.experiments.text.common import vocab
    from axlearn.common.config import config_for_function
    # model_id = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_to_transformers/Llama-2-7b-hf"
    model_id = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_to_transformers/baseline_34000"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(model_id)
    model.to(device)
    sentencepiece_model_name = "bpe_32k_c4.model"
    vocab_cfg = config_for_function(vocab).set(
        sentencepiece_model_name=sentencepiece_model_name, num_extra_ids=None
    )
    tokenizer = vocab_cfg.instantiate()
    generation_config = GenerationConfig(
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        do_sample=True,
        temperature=0.6,
        max_length=4096,
        top_p=0.9
    )
    for text in texts:
        token_ids = tokenizer.encode(text)
        import pdb; pdb.set_trace()
        token_ids.insert(0, tokenizer.eos_id)
        inputs = {
            # add a eos_token_id to be consistent with training
            "input_ids": torch.tensor([token_ids]).to(device),
            "attention_mask": torch.ones(1, len(token_ids), dtype=torch.int).to(device),
        }
        outputs = model.generate(**inputs, generation_config=generation_config)
        output_texts = tokenizer.decode(outputs[0])
        print(output_texts)


# model_inputs in transformers/pipelines/base.py(1261)
# {'input_ids': tensor([[128000,  15339,   1917,    198]]), 'attention_mask': tensor([[1, 1, 1, 1]]), 'prompt_text': 'hello world\n'}
texts = [
    "How are you doing?",
    "who is the president of the US now?",
    "The USA is in which continent?",
    "California is a state in",
    "Can you tell me something about California state?\n",
    "California is a state in",
]
# run_pipeline(texts, run_original=False)
# run_pipeline(texts)
run_fuji_model(texts)
# run_model(texts)

texts = [
    "new twerk team twerk songs 2010 twerk",
    # new twerk team twerk songs 2010 twerk booty twerk team youtube twerking on boy twerk team dance girl twerking twerk i | Twerking pics « twerk team video twerk ass twerking dance twerk urban dictionary twerk shirts twerk team dancing twerk team ass twerk songs urband ictionar new twerk team twerk team exposed girls twerking twerk dancing twerk team se » Posted in Ass shake | Tags: ass twerkintwerk parties, big booty twertwerk twerk, booty twer, ebony twertwerk dance, twerk team pics, twerk team uncensoretwerk videotwerking definition, twerk team videobooty twerkinbooty twerk
    "Korean cosmetic teeth 07090 Tucson",
    # Korean cosmetic teeth 07090 Tucson whitening toothpaste lumineers veneers 07716 what are veneers for teeth 10601 porcelain crowns cost in Danbury Connecticut Gay dental emergency Fenton MO zoom teeth whitening in Port Townsend WA
    # "State of Art blazer",
    "Volvo Power Steering",
    # Volvo Power Steering Pressure Hose (C70 S70 V90) Genuine Volvo 9485359 | FCP Euro Review specifics Volvo C70 1999 Volvo C70 Power Steering Hose 2000 Volvo C70 Power Steering Hose 2002 Volvo C70 Power Steering Hose 2003 Volvo C70 Power Steering Hose 2004 Volvo C70 Power Steering Hose Review specifics Volvo S70 1999 Volvo S70 Power Steering Hose 
    "Free store pickupas",
    # Free store pickupas soon as 9/04with •Free store pickupas soon as 9/04with The Pajama Game (1957) (Full Frame) Free store pickupas soon 
]

run_fuji_model(texts)
# run_model(texts)
