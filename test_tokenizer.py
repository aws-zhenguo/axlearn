# import tensorflow as tf
# import seqio
# 
# tokenizer_path = "/fsx/czhenguo/Projects/fruitstand/axlearn/axlearn/data/tokenizers/sentencepiece/bpe_32k_c4.model"
# tokenizer = seqio.SentencePieceVocabulary(sentencepiece_model_file=tokenizer_path)
# 
# 
# # tf_checkpoint = tf.train.Checkpoint(tokenizer=tokenizer)
# # tf_checkpoint.write("./saved_iter.ckpt")
# 
# #   with open("/path/to/tokenizer.model", "rb") as f:
# #       sp_model = f.read()
# #   tokenizer = tf_text.SentencepieceTokenizer(sp_model)
# ds = tf.data.Dataset.from_tensor_slices(dict(data=["ex1", "ex2", "ex3",]))
# 
# def _map(ex):
#     # print(ex)
#     # def tokenize_fn(text):
#     #     return tokenizer.encode_tf(text)
# 
#     # return dict(data=tf.py_function(func=tokenize_fn, inp=ex["data"], Tout=tf.int32))
#     return dict(data=tokenizer.encode_tf(ex["data"]))
# 
# import pdb; pdb.set_trace()
# ds: tf.data.Dataset = ds.map(_map)
# iterator = iter(ds)
# ckpt = tf.train.Checkpoint(iterator=iterator)
# ckpt.write("/tmp/iterator")


import sentencepiece as spm
from transformers import convert_slow_tokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer
# from transformers.convert_slow_tokenizer import SpmConverter

def convert_tokenizer():
    tokenizer_path = "axlearn/data/tokenizers/sentencepiece/bpe_32k_c4.model"
    spm_tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    # spm_tokenizer.vocab_file = 'axlearn/data/tokenizers/sentencepiece/bpe_32k_c4.vocab'
    spm_tokenizer.vocab_file = tokenizer_path
    # spm_converter = SpmConverter(spm_tokenizer)
    spm_converter = convert_slow_tokenizer.SpmConverter(spm_tokenizer)
    # import pdb; pdb.set_trace()
    converted = spm_converter.converted()
    converted.save('converted.json')
    
    import pdb; pdb.set_trace()
    tok = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path='converted.json', clean_up_tokenization_spaces=True, pad_token='<pad>', unk_token='<unk>', bos_token='<s>', eos_token='</s>', model_max_length=1024, padding_side='right', truncation_side='right')
    # tok = PreTrainedTokenizerFast(tokenizer_object=converted, clean_up_tokenization_spaces=False, pad_token='<pad>', unk_token='<unk>', bos_token='<s>', eos_token='</s>', model_max_length=1024, padding_side='right', truncation_side='right')
    # tok = PreTrainedTokenizer.from_pretrained(pretrained_model_name_or_path='converted.json', clean_up_tokenization_spaces=False, pad_token='<pad>', unk_token='<unk>', bos_token='<s>', eos_token='</s>', model_max_length=1024, padding_side='right', truncation_side='right')
    # tok = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path='converted.json', clean_up_tokenization_spaces=False, pad_token_id=spm_tokenizer.pad_id, unk_token_id=spm_tokenizer.unk_id, bos_token_id=spm_tokenizer.bos_id, eos_token_id=spm_tokenizer.eos_id, model_max_length=1024, padding_side='right', truncation_side='right')
    # import pdb; pdb.set_trace()
    tok.save_pretrained('ConvertedTokenizer')

def run_tokenizer(texts):
    tokenizer = PreTrainedTokenizerFast.from_pretrained('ConvertedTokenizer')
    import pdb; pdb.set_trace()
    for text in texts:
        tokenizer.tokenize(text)
    return tokenizer.batch_encode_plus(texts)

def run_sentence_piece_tokenizer(texts):
    tokenizer_path = "axlearn/data/tokenizers/sentencepiece/bpe_32k_c4.model"
    spm_tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

    import pdb; pdb.set_trace()
    token_ids = spm_tokenizer.encode(texts)
    import pdb; pdb.set_trace()
    return token_ids

def run_seqio_tokenizer(texts):
    from axlearn.experiments.text.common import vocab
    from axlearn.common.config import config_for_function
    sentencepiece_model_name = "bpe_32k_c4.model"
    vocab_cfg = config_for_function(vocab).set(
        sentencepiece_model_name=sentencepiece_model_name, num_extra_ids=None
    )
    tokenizer = vocab_cfg.instantiate()
    import pdb; pdb.set_trace()
    token_ids = tokenizer.encode(texts)
    print(token_ids)
    import pdb; pdb.set_trace()
    return token_ids

def compare_tokenizers():
    from axlearn.experiments.text.common import vocab
    from axlearn.common.config import config_for_function
    from axlearn.experiments.text.gpt import c4_trainer
    # converted_tokenizer = PreTrainedTokenizerFast.from_pretrained('ConvertedTokenizer')
    converted_tokenizer = PreTrainedTokenizer.from_pretrained('ConvertedTokenizer')
    # sentencepiece_model_name = "bpe_32k_c4.model"
    # vocab_cfg = config_for_function(vocab).set(
    #     sentencepiece_model_name=sentencepiece_model_name, num_extra_ids=None
    # )
    # seqio_tokenizer = vocab_cfg.instantiate()
    fuji_model_name = "fuji-7B-v2"
    trainer_config_map = c4_trainer.named_trainer_configs()
    trainer_config_fn = trainer_config_map[fuji_model_name]
    trainer_config = trainer_config_fn()
    tokenizer_cfg = trainer_config.input.source.vocab_cfg
    seqio_tokenizer = tokenizer_cfg.instantiate()
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    convert_tokenizer()
    texts = [
        "How are you doing?",
        "who is the president of the US now?",
        "The USA is in which continent?",
        "California is a state in",
        "Can you tell me something about California state?\n",
        "California is a state in",
    ]
    # print(run_tokenizer(texts))
    print(run_sentence_piece_tokenizer(texts))
    # print(run_seqio_tokenizer(texts))
    # compare_tokenizers()
