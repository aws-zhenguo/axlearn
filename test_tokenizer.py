import sentencepiece as spm
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer, AutoTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer

# sentencepiece_tokenizer_path = "/fsx/czhenguo/Projects/fruitstand/axlearn/axlearn/data/tokenizers/sentencepiece/bpe_32k_c4.model"
# sentencepiece_tokenizer_path = "/fsx/czhenguo/Projects/fruitstand/axlearn/axlearn/data/tokenizers/sentencepiece/bpe_32k_c4.v2model"
sentencepiece_tokenizer_path = "/fsx/czhenguo/Projects/fruitstand/axlearn/axlearn/data/tokenizers/sentencepiece/bpe_32k_c4.v1model"

def convert_tokenizer_transformers_4_47_1():
    from transformers.convert_slow_tokenizer import SpmConverter

    spm_tokenizer = spm.SentencePieceProcessor(model_file=sentencepiece_tokenizer_path)
    spm_tokenizer.vocab_file = sentencepiece_tokenizer_path
    spm_converter = SpmConverter(spm_tokenizer)
    converted = spm_converter.converted()
    converted.save('converted.json')
    print(converted.encode("How are you doing?").tokens)
    import pdb; pdb.set_trace()

    tok = PreTrainedTokenizer(tokenizer_object=converted, use_fast=False, clean_up_tokenization_spaces=False, pad_token='<pad>', unk_token='<unk>', bos_token='<s>', eos_token='</s>', model_max_length=1024, padding_side='right', truncation_side='right')
    tok.save_pretrained('ConvertedTokenizer')

def convert_tokenizer_transformers_4_43_2():
    from transformers import convert_slow_tokenizer

    spm_tokenizer = spm.SentencePieceProcessor(model_file=sentencepiece_tokenizer_path)
    spm_tokenizer.vocab_file = sentencepiece_tokenizer_path
    spm_tokenizer.vocab_file = tokenizer_path
    spm_converter = convert_slow_tokenizer.SpmConverter(spm_tokenizer)
    # import pdb; pdb.set_trace()
    converted = spm_converter.converted()
    converted.save('converted.json')
    
    tok = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path='converted.json', clean_up_tokenization_spaces=True, pad_token='<pad>', unk_token='<unk>', bos_token='<s>', eos_token='</s>', model_max_length=1024, padding_side='right', truncation_side='right')
    tok.save_pretrained('ConvertedTokenizer')

def run_tokenizer(texts):
    tokenizer = AutoTokenizer.from_pretrained('ConvertedTokenizer', use_fast=False)
    for text in texts:
        print(tokenizer.tokenize(text))
        print(tokenizer.encode(text))
    return tokenizer.batch_encode_plus(texts)

def run_sentence_piece_tokenizer(texts):
    tokenizer_path = sentencepiece_tokenizer_path
    spm_tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

    import pdb; pdb.set_trace()
    token_ids = spm_tokenizer.encode(texts)
    return token_ids

def run_seqio_tokenizer(texts):
    from axlearn.experiments.text.common import vocab
    from axlearn.common.config import config_for_function

    sentencepiece_model_name = "bpe_32k_c4.model"
    vocab_cfg = config_for_function(vocab).set(
        sentencepiece_model_name=sentencepiece_model_name, num_extra_ids=None
    )
    tokenizer = vocab_cfg.instantiate()
    token_ids = tokenizer.encode(texts)
    return token_ids

def run_c4_tokenizer(texts):
    from axlearn.experiments.text.common import vocab
    from axlearn.common.config import config_for_function
    from axlearn.experiments.text.gpt import c4_trainer

    fuji_model_name = "fuji-7B-v2"
    trainer_config_map = c4_trainer.named_trainer_configs()
    trainer_config_fn = trainer_config_map[fuji_model_name]
    trainer_config = trainer_config_fn()
    tokenizer_cfg = trainer_config.input.source.vocab_cfg
    tokenizer = tokenizer_cfg.instantiate()
    token_ids = tokenizer.encode(texts)
    return token_ids

def run_mistral_tokenizer(texts):
    tokenizer_path = sentencepiece_tokenizer_path
    tokenizer = SentencePieceTokenizer(model_path=tokenizer_path)
    for text in texts:
        token_ids = tokenizer.encode(text, bos=True, eos=False)
        print(token_ids)
    import pdb; pdb.set_trace()
    return token_ids

if __name__ == "__main__":
    # convert_tokenizer()
    # import tokenizers
    # tok = tokenizers.SentencePieceTokenizer(sentencepiece_tokenizer_path)
    # convert_tokenizer_transformers_4_47_1()
    texts = [
        "How are you doing?",
        "who is the president of the US now?",
        "The USA is in which continent?",
        "California is a state in",
        "Can you tell me something about California state?\n",
        "California is a state in",
    ]
    # print(run_tokenizer(texts))
    # print(run_sentence_piece_tokenizer(texts))
    print(run_mistral_tokenizer(texts))
    # print(run_seqio_tokenizer(texts))
    # compare_tokenizers()
