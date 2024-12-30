import jax
import numpy as np
import seqio
import tensorflow as tf
import torch
from transformers import AutoConfig, LlamaForCausalLM

from collections import defaultdict
from jax import numpy as jnp
from axlearn.common import config, evaler, input_tf_data, measurement, utils
from axlearn.common.checkpointer import Checkpointer
from axlearn.common.config import config_for_function
from axlearn.common.decoder import LmHead
from axlearn.common.decoding import StopOnSubsequence
from axlearn.common.inference import InferenceRunner
from axlearn.common.inference_pipeline import pop_string_tensors
from axlearn.common.input_lm import lm_text_preprocessor, text2text_lm_input, text_to_lm_eval_input
from axlearn.common.module import functional
from axlearn.experiments import get_named_trainer_config
from axlearn.experiments.text.common import vocab
from axlearn.experiments.text.gpt import c4_trainer
from axlearn.vision import image_classification, input_image, resnet
from utils import (
    seed,
    get_fuji_and_llama,
    load_checkpoint,
    get_mesh,
    parameters_from_llama,
    parameters_to_llama,
    save_axlearn_checkpoint,
    save_transformers_checkpoint,
    copy_tokenizer_files,
    init_infer_runner,
)

# config_name = "fuji-1B-v3"
# config_name = "fuji-7B-v2"
config_name = "fuji-70B-v2"

# Note: step folder need to be included for inference runner but not for checkpointer
# to specify step folder for checkpointer, use the step parameter
# CHECKPOINT_PATH = "/fsx/thiamha/fs/runs/artifacts/fs_main2/tg6.7b/241122003124/axlearn_out/checkpoints"
if config_name == "fuji-1B-v3":
    CHECKPOINT_PATH = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/test_01/11132/axlearn_out/checkpoints"
    # CHECKPOINT_PATH = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/test_01/11130/axlearn_out/checkpoints"
    sentencepiece_model_name = "bpe_128k_c4.model"
elif config_name == "fuji-7B-v2":
    CHECKPOINT_PATH = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10976/axlearn_out/checkpoints/step_00034000"
    # CHECKPOINT_PATH = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10976/axlearn_out/checkpoints"
    # CHECKPOINT_PATH = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/transformers_to_axlearn/fuji-7B-v2/step_00022794"
    # CHECKPOINT_PATH = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/transformers_to_axlearn/round_trip/step_00022794"
    sentencepiece_model_name = "bpe_32k_c4.model"
    converted_tokenizer_path = "ConvertedTokenizer"
else:
    # CHECKPOINT_PATH = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/trn_baselines/611/axlearn_out/checkpoints/step_00010000"
    CHECKPOINT_PATH = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10985/axlearn_out/checkpoints/step_00035000"
    sentencepiece_model_name = "bpe_32k_c4.model"
    converted_tokenizer_path = "ConvertedTokenizer"

trainer_config_fn = get_named_trainer_config(
    config_name, config_module="axlearn.experiments.text.gpt.c4_trainer"
)
# TODO trainer_config connot be removed for now since mesh is needed to save checkpoint
trainer_config = trainer_config_fn()
use_transformers = False


def get_transformers_tokenizer():
    """Replace sentence piece tokenizer with transformers tokenizer when loading Llama checkpoints."""
    from transformers import AutoTokenizer

    if config_name == "fuji-1B-v3":
        tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-1B")
    else:
        tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b-hf")
    return tokenizer


def make_ds_fn(
    is_training: bool, texts: list[str], repeat: int = 1
) -> input_tf_data.BuildDatasetFn:
    del is_training

    def ds_fn() -> tf.data.Dataset:
        def data_gen():
            for _ in range(repeat):
                for index, text in enumerate(texts):
                    yield {"text": text, "index": index}

        return tf.data.Dataset.from_generator(
            data_gen,
            output_signature={
                "text": tf.TensorSpec(shape=(), dtype=tf.string),
                "index": tf.TensorSpec(shape=(), dtype=tf.uint32),
            },
        )

    return ds_fn


def run_inference(texts):
    infer_runner, infer_runner_config = init_infer_runner(trainer_config, CHECKPOINT_PATH)
    mesh = get_mesh(trainer_config)
    model = infer_runner.model
    evaler_config = trainer_config.evalers["validation"]
    evaler_config.name = "validation"
    evaler_config.summary_writer.dir = "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/ttest_01/11126/axlearn_out/summaries/validation"

    # init tokenizer for decode
    vocab_cfg = config_for_function(vocab).set(
        sentencepiece_model_name=sentencepiece_model_name, num_extra_ids=None
    )
    sentence_piece_vocab = vocab_cfg.instantiate()

    batch_size, max_len = 8, 4096
    evaler_config.input = evaler_config.input.set(
        source=config_for_function(make_ds_fn).set(texts=texts),
        processor=config_for_function(text_to_lm_eval_input).set(
            vocab_cfg=vocab_cfg,
            max_len=max_len,
            replace_newlines_with="\n",
            stride=2,
        ),
        batcher=evaler_config.input.batcher.set(global_batch_size=batch_size),
    )

    results = list()

    with mesh:
        model_param_specs = model.create_parameter_specs_recursively()
        model_param_partition_specs = jax.tree.map(lambda spec: spec.mesh_axes, model_param_specs)
        evaler = evaler_config.instantiate(
            parent=None,
            model=model,
            model_param_partition_specs=model_param_partition_specs,
        )
        eval_input_iter = iter(evaler.input.dataset())
        prng_key = jax.random.PRNGKey(seed=seed)
        method_runner = infer_runner.create_method_runner(method="predict", prng_key=prng_key)

        for batch_ix, input_batch in enumerate(evaler.input.batches(eval_input_iter)):
            import pdb; pdb.set_trace()
            input_ids = input_batch["input_ids"].tolist()
            input_texts = sentence_piece_vocab.tokenizer.decode_ids(input_ids)
            input_batch, input_batch_str_tensors = pop_string_tensors(input_batch)
            input_batch = utils.as_numpy_array(input_batch)
            global_input_batch = utils.host_to_global_device_array(
                input_batch, partition=infer_runner_config.input_batch_partition_spec
            )
            output = method_runner(global_input_batch)
            output_batch = utils.global_to_host_array(
                output.output_batch,
                partition=infer_runner_config.input_batch_partition_spec,
            )
            output_batch = utils.global_to_host_array(
                output.output_batch,
                partition=infer_runner_config.input_batch_partition_spec,
            )
            # (16, 4096, 32768)
            logits = output.output_batch["logits"]
            output_ids = jax.numpy.argmax(logits, axis=-1)
            output_texts = sentence_piece_vocab.tokenizer.decode_ids(output_ids.tolist())
            # sentence_piece_vocab.tokenizer.pad_id()  # 0
            # sentence_piece_vocab.tokenizer.eos_id()  # 1
            # sentence_piece_vocab.tokenizer.bos_id()  # -1

            results.extend(output_texts)
            print(output_texts)
    return results


def get_version(model_name):
    if "v3" in model_name:
        return 3
    else:
        return 2


def get_sentence_piece_tokenizer():
    vocab_cfg = config_for_function(vocab).set(
        sentencepiece_model_name=sentencepiece_model_name, num_extra_ids=None
    )
    sentence_piece_vocab = vocab_cfg.instantiate()
    return sentence_piece_vocab


def relative_difference(x, y):
    # May get hugh relative difference since some logits would be super small
    return np.abs(x - y) / np.min([np.abs(x), np.abs(y)], axis=0)

def average_top_k_jaccard_similarity(x, y, max_k=100):
    ks = [i for i in range(1, max_k + 1, 5)]
    jaccard_indices = defaultdict(int)
    batch_size, seq_len, vocab_size = x.shape

    for x_sequence, y_sequence in zip(x, y):
        for x_logits, y_logits in zip(x_sequence, y_sequence):
            for k in ks:
                top_x_indices = x_logits.argsort()[::-1]
                top_y_indices = y_logits.argsort()[::-1]
                top_x_set = set(top_x_indices[:k])
                top_y_set = set(top_y_indices[:k])
                jaccard_index = len(top_x_set.intersection(top_y_set)) / len(top_x_set.union(top_y_set))
                jaccard_indices[k] += jaccard_index

    for k in jaccard_indices:
        jaccard_indices[k] /= (batch_size * seq_len)
    return jaccard_indices

def relative_difference_with_top_p(x, y):
    return None


def get_top_p(x, p):
    total = 0.0
    top_values = list()
    for val in x:
        total += val
        top_values.append(val)
        if total > p:
            break
    return top_values


def get_top_p_indices(x, x_indices, p):
    total = 0.0
    top_p_indices = list()
    for idx in x_indices:
        total += x[idx]
        top_p_indices.append(idx)
        if total > p:
            break
    return top_p_indices


def ndarray_apply_order(x, indices):
    # x[indices] does not work for ndarray with dimension higher than 2
    # since different indices need to be applied on different array
    result = list()
    for i, seq in enumerate(x):
        new_seq = list()
        for j, tokens in enumerate(seq):
            new_seq.append(tokens[indices[i][j]])
        result.append(new_seq)
    return np.asarray(result)


def assert_top_p_allclose(x, y, atol, rtol, p=0.99):
    # x = np.sort(x, axis=-1)[:,:,::-1]
    # y = np.sort(y, axis=-1)[:,:,::-1]
    x_indices = np.argsort(x, axis=-1)[:,:,::-1]

    top_x_flat = list()
    top_y_flat = list()
    for i, (x_seq, y_seq) in enumerate(zip(x, y)):
        for j, (x_token, y_token) in enumerate(zip(x_seq, y_seq)):
            top_p_indices = get_top_p_indices(x_token, x_indices[i][j], p)
            top_x = x_token[top_p_indices]
            top_y = y_token[top_p_indices]

            top_x_flat.extend(top_x)
            top_y_flat.extend(top_y)

    top_x_flat = np.asarray(top_x_flat)
    top_y_flat = np.asarray(top_y_flat)
    rdiff = relative_difference(top_x_flat, top_y_flat)
    print("max top p difference", np.max(top_x_flat - top_y_flat))
    print("mean top p difference", np.mean(top_x_flat - top_y_flat))
    print("max top p relative difference", np.max(rdiff))
    print("mean top p relative difference", np.mean(rdiff))

    np.testing.assert_allclose(top_x, top_y, atol=atol, rtol=rtol, equal_nan=False)


def assert_top_k_allclose(x, y, atol, rtol, k=64):
    x_indices = np.argsort(x, axis=-1)[:,:,::-1]
    top_x = ndarray_apply_order(x, x_indices)[:,:,:k]
    top_y = ndarray_apply_order(y, x_indices)[:,:,:k]
    # x = np.sort(x, axis=-1)[:,:,::-1]
    # y = np.sort(y, axis=-1)[:,:,::-1]

    rdiff = relative_difference(top_x, top_y)
    print("max top k difference", np.max(top_x - top_y))
    print("mean top k difference", np.mean(top_x - top_y))
    print("max top k relative difference", np.max(rdiff))
    print("mean top k relative difference", np.mean(rdiff))

    np.testing.assert_allclose(top_x, top_y, atol=atol, rtol=rtol, equal_nan=False)


def assert_top_p_and_top_k_allclose(x, y, atol, rtol, k=64, p=0.99):
    x_indices = np.argsort(x, axis=-1)[:,:,::-1]

    top_x_flat = list()
    top_y_flat = list()
    for i, (x_seq, y_seq) in enumerate(zip(x, y)):
        for j, (x_token, y_token) in enumerate(zip(x_seq, y_seq)):
            top_indices = list()
            total = 0.0
            for idx in x_indices[i][j]:
                total += x_token[idx]
                top_indices.append(idx)

                if total > p and len(top_indices) >= k:
                    break

            top_x = x_token[top_indices]
            top_y = y_token[top_indices]

            top_x_flat.extend(top_x)
            top_y_flat.extend(top_y)

    top_x_flat = np.asarray(top_x_flat)
    top_y_flat = np.asarray(top_y_flat)
    rdiff = relative_difference(top_x_flat, top_y_flat)
    print("max top p/k difference", np.max(top_x_flat - top_y_flat))
    print("mean top p/k difference", np.mean(top_x_flat - top_y_flat))
    print("max top p/k relative difference", np.max(rdiff))
    print("mean top p/k relative difference", np.mean(rdiff))

    np.testing.assert_allclose(top_x, top_y, atol=atol, rtol=rtol, equal_nan=False)


def validate_conversion(fuji_model_name, llama_model_name, load_true_model=False, reverse=False):
    """Run a forward pass and compare logits to validate conversion between fuji and llama model."""
    fuji, state, llama = get_fuji_and_llama(
        fuji_model_name, llama_model_name, load_true_model, reverse, fuji_model_path=CHECKPOINT_PATH
    )

    # generate dummy input data
    # ids = jax.random.randint(jax.random.PRNGKey(seed), shape=(2, 2), minval=0, maxval=12345)
    # end to end test with texts
    tokenizer = get_sentence_piece_tokenizer()
    def pad_list(l, max_len, fill_value=tokenizer.pad_id):
        return [tokenizer.eos_id] + l + [fill_value] * (max_len - len(l) - 1)

    texts = [
        "How are you doing?",
        "who is the president of the US now?",
        "The USA is in which continent?",
        "California is a state in",
        "Can you tell me something about California state?\n",
    ]
    ids = tokenizer.encode(texts)
    padded_ids = [pad_list(cur_ids, 6) for cur_ids in ids]
    ids = [cur_ids[:4] for cur_ids in padded_ids]
    target_ids = [cur_ids[1:5] for cur_ids in padded_ids]
    torch_ids = torch.from_numpy(np.asarray(ids))
    torch_target_ids = torch.from_numpy(np.asarray(target_ids))

    # conversion for llama2 and llama3 would be different
    # for example llama3 would use GQA and some of the model also share weights between lm_head and emb
    if "70B" in fuji_model_name:
        use_gqa = True
    else:
        use_gqa = False

    # convert params
    if reverse:
        llama_state_dict = parameters_to_llama(state, llama, use_gqa)
        llama.load_state_dict(llama_state_dict)
    else:
        state = parameters_from_llama(llama, state, use_gqa)

    input_batch = {"input_ids": jnp.asarray(ids), "target_labels": jnp.asarray(target_ids)}
    # (_, aux), _ = functional(
    (loss, aux), output_collection = functional(
        fuji,
        is_training=False,
        prng_key=jax.random.PRNGKey(seed),
        state=state,
        inputs={"input_batch": input_batch, "return_aux": True},
    )

    with torch.no_grad():
        output = llama(torch_ids, labels=torch_target_ids)
        # transformers will shift the ids when calculating the loss
        # so to make sure the loss would match, adjust the input
        # https://github.com/huggingface/transformers/blob/641adca55832ed9c5648f54dcd8926d67d3511db/src/transformers/models/llama/modeling_llama.py#L833
        extra_ids = [cur_ids[:5] for cur_ids in padded_ids]
        extra_ids = torch.from_numpy(np.asarray(extra_ids))
        output_with_loss = llama(extra_ids, labels=extra_ids)
        llama_loss = output_with_loss.loss

    fuji_logits = np.asarray(aux["logits"])
    llama_logits = output.logits.numpy()
    # rdiff = relative_difference(fuji_logits, llama_logits)
    # jaccard_indices = average_top_k_jaccard_similarity(fuji_logits, llama_logits)
    assert isinstance(aux["logits"].dtype, np.dtypes.Float32DType)

    # The difference is caused by the SDPA attention layer. The deeper the larger the error.
    if fuji_model_name == "fuji-1B-v3":
        atol = 3e-2
        rtol = 1e-4
    elif fuji_model_name == "fuji-3B-v3":
        atol = 2e-2
        rtol = 1e-4
    elif fuji_model_name == "fuji-7B-v2":
        atol = 1e-3
        rtol = 1e-4
    elif fuji_model_name == "fuji-8B-v3":
        atol = 2e-1
        rtol = 1e-4
    elif fuji_model_name == "fuji-70B-v3":
        atol = 2.0
        rtol = 1e-4
    else:
        atol = 2e-3
        rtol = 1e-4

    import pdb; pdb.set_trace()
    # TODO should not do the softmax myself
    fuji_probs = np.asarray(jax.nn.softmax(aux["logits"]))
    llama_probs = torch.softmax(output.logits, dim=-1).numpy()
    assert isinstance(fuji_probs.dtype, np.dtypes.Float32DType)

    assert_top_k_allclose(llama_probs, fuji_probs, atol, rtol)
    assert_top_p_allclose(llama_probs, fuji_probs, atol, rtol)
    assert_top_p_and_top_k_allclose(llama_probs, fuji_probs, atol, rtol)
    print("smallest fuji prob:", np.min(fuji_probs))
    print("smallest llama prob:", np.min(llama_probs))

    np.save("fuji_probs", fuji_probs)
    np.save("llama_probs", llama_probs)
    # np.load("fuji_probs.npy")


    # np.testing.assert_allclose(fuji_logits, llama_logits, atol=atol, equal_nan=False)


def convert_and_save_checkpoint(
    fuji_model_name, llama_model_name, load_true_model=True, reverse=False, fuji_model_path=None, save_name="converted_model"
):
    fuji, state, llama = get_fuji_and_llama(
        fuji_model_name, llama_model_name, load_true_model=load_true_model, reverse=reverse, fuji_model_path=fuji_model_path
    )
    if "70B" in fuji_model_name:
        use_gqa = True
    else:
        use_gqa = False

    # convert params
    if reverse:
        llama_state_dict = parameters_to_llama(
            state, llama, use_gqa
        )
        llama.load_state_dict(llama_state_dict)
        checkpoint_path = f"/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_to_transformers/{save_name}"
        save_transformers_checkpoint(llama, checkpoint_path)
        # Seems spm and transformers tokenizers are not matching
        # copy_tokenizer_files(converted_tokenizer_path, checkpoint_path)
        # Remember to copy the generation_config.json to checkpoint folder to get similar results
    else:
        state = parameters_from_llama(llama, state, use_gqa)
        checkpoint_path = f"/fsx/czhenguo/Projects/fruitstand/runs/artifacts/transformers_to_axlearn/{save_name}"

        save_axlearn_checkpoint(fuji, state, checkpoint_path, get_mesh(trainer_config))

    print(f"Successfuly save checkpoint to {checkpoint_path}.")


def get_fuji_with_llama_weights():
    model_config = trainer_config.model
    model_config.set(name="fuji-test-model")

    if config_name == "fuji-7B-v2":
        # llama2 7B does not share lm_head with embedding, but fuji does
        # need to disable lm_head sharing for fuji to match llama
        model_config.decoder.set(lm_head=LmHead.default_config())

    llama = LlamaForCausalLM.from_pretrained("Llama-2-7b-hf", local_files_only=True)
    prng_key = jax.random.PRNGKey(seed)
    model = model_config.instantiate(parent=None)
    model_state = model.initialize_parameters_recursively(prng_key=prng_key)
    model_state = parameters_from_llama(llama, model_state, 2)
    return model, model_state


def generate(texts):
    # TODO init model and load checkpoint without InferenceRunner
    # model = infer_runner.model
    # model_state = infer_runner._inference_runner_state.model
    results = list()
    batch_size, max_len = 8, 4096

    # init tokenizer for decode
    if use_transformers:
        # model, model_state = get_fuji_with_llama_weights()
        model_config = trainer_config.model
        model_config.set(name="model")
        # TODO remove the following two lines
        # model_config.decoder.set(lm_head=LmHead.default_config())
        # model_config.decoder.set(vocab_size=32000)

        model = model_config.instantiate(parent=None)
        model_state = load_checkpoint(trainer_config, CHECKPOINT_PATH)

        # update pad_token_id since they are different in fuji and llama
        tokenizer = get_transformers_tokenizer()
        pad_token_id = model.config.decoder.pad_token_id
        tokenizer.pad_token_id = pad_token_id
        eos_token_id = tokenizer.eos_token_id

        # Fuji models use different eos_token_id than llama models
        # fuji eos_token_id = 0, llama eos_token_id = 128001 (llama3.2 1B)
        stop_decoding_condition = StopOnSubsequence([[tokenizer.eos_token_id]])

        input_ids = tokenizer.batch_encode_plus(texts, padding="max_length", max_length=max_len)[
            "input_ids"
        ]
    else:
        model_config = trainer_config.model
        model_config.set(name="model")
        # TODO remove the following two lines
        # model_config.decoder.set(lm_head=LmHead.default_config())
        # model_config.decoder.set(lm_head=None)

        model = model_config.instantiate(parent=None)
        model_state = load_checkpoint(trainer_config, CHECKPOINT_PATH)

        vocab_cfg = config_for_function(vocab).set(
            sentencepiece_model_name=sentencepiece_model_name, num_extra_ids=None
        )
        tokenizer = vocab_cfg.instantiate()
        eos_token_id = tokenizer.eos_id
        stop_decoding_condition = StopOnSubsequence([[model.decoder.config.eos_token_id]])

        input_ids = tokenizer.encode(texts)
        # TODO the inference for fuji model seem to be broken

        # add padding
        def pad_list(l, max_len, fill_value=tokenizer.pad_id):
            # need to add a eos_token_id at the begining to be consistent with text_to_lm_eval_input
            return [eos_token_id] + l + [fill_value] * (max_len - len(l) - 1)

        input_ids = [pad_list(ids, max_len) for ids in input_ids]
        # input_ids = np.pad(input_ids, ((0, 0),(0, max_len)), "constant", constant_values=0)

    method = "sample_decode"
    # method="beam_search_decode"
    # import pdb; pdb.set_trace()

    # TODO decoder batch mode
    for input_id in input_ids:
        # follow decoder input format https://github.com/apple/axlearn/blob/a15a3bcbb976c14db157a8958df368a48c614c1f/axlearn/common/decoder_test.py#L569
        input_batch = {
            "input_batch": {"prefix": jax.numpy.asarray([input_id])},
        }
        # Override the default decoder eos_token_id since fuji and llama has different eos_token_id
        # but beam_search_decode does not accept this argument
        # https://github.com/apple/axlearn/blob/a15a3bcbb976c14db157a8958df368a48c614c1f/axlearn/common/decoder.py#L336
        if method == "sample_decode":
            input_batch["stop_decoding_condition"] = stop_decoding_condition
        # TODO add mask for batch running https://github.com/apple/axlearn/blob/a15a3bcbb976c14db157a8958df368a48c614c1f/axlearn/common/decoder_test.py#L563C13-L563C24

        # TODO how to get model states with invocation context without using infer_runner? https://github.com/apple/axlearn/blob/a15a3bcbb976c14db157a8958df368a48c614c1f/axlearn/experiments/text/gpt/param_converter_test.py#L105
        # TODO the decoding process does not seem to start from the last token, but start from the first token when testing with axlearn model
        output, _ = functional(
            model,
            is_training=False,
            prng_key=jax.random.PRNGKey(seed=seed),
            state=model_state,
            inputs=input_batch,
            method=method,
        )
        # need to manually remove tokens after eos_token if using sample_decode
        # because it will call _decode_init and create a sequence with the length max_len
        # https://github.com/apple/axlearn/blob/main/axlearn/common/decoding.py#L790
        batch, num, indices = jax.numpy.where(output.sequences == eos_token_id)
        if use_transformers:
            decode_fn = tokenizer.batch_decode
        else:
            decode_fn = lambda x: tokenizer.decode(x[0])

        if indices.size > 0:
            output_texts = decode_fn([output.sequences[0][0][: indices[0]]])
        else:
            # in case eos_token is not generated
            output_texts = decode_fn(output.sequences[0][0])
        import pdb; pdb.set_trace()
        print(output_texts)
        results.extend([output_texts])
    return results


if __name__ == "__main__":
    texts = [
        "How are you doing?",
        "who is the president of the US now?",
        "The USA is in which continent?",
        "California is a state in",
        "Can you tell me something about California state?\n",
        "California is a state in",
        "California is a state in",
        "California is a state in",
    ]
    # run_inference(texts)
    # generate(texts)
    # validate_conversion("fuji-1B-v3", "Llama-3.2-1B", load_true_model=True)
    # validate_conversion("fuji-7B-v2", "Llama-2-7b-hf", load_true_model=True)
    # validate_conversion("fuji-7B-v2", "Llama-2-7b-hf", load_true_model=False)
    # validate_conversion("fuji-7B-v2", "Llama-2-7b-hf", load_true_model=True, reverse=True)
    validate_conversion("fuji-70B-v2", "Llama-2-70b-hf", load_true_model=True, reverse=True)
    # convert_and_save_checkpoint("fuji-7B-v2", "Llama-2-7b-hf", load_true_model=True, reverse=False)
    # convert_and_save_checkpoint("fuji-7B-v2", "/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_to_transformers/baseline_34000", load_true_model=True, reverse=False, save_name="round_trip")
    # convert_and_save_checkpoint("fuji-7B-v2", "Llama-2-7b-hf", load_true_model=True, reverse=True, fuji_model_path=CHECKPOINT_PATH, save_name="baseline_34000")
    # convert_and_save_checkpoint("fuji-7B-v2", "Llama-2-7b-hf", load_true_model=False, reverse=True, save_name="random_init")
