import numbers
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Callable, NamedTuple, Optional, TypeVar, Union

import jax
import numpy as np
import torch
from jax import numpy as jnp
from transformers import AutoConfig, LlamaForCausalLM

from axlearn.common.decoder import LmHead
from axlearn.experiments.text.gpt import c4_trainer


def as_jax_tensor(x: Any):
    """Converts `x` to jax.Array recursively.

    Args:
        x: a jnp array, numpy array, TF/PyTorch Tensor, or a nested structure of arrays or Tensors.

    Returns:
        A nested structure with the same structure as `x` but with values converted to jax.Arrays.

    Raises:
        NotImplementedError: If conversion for the input type is unsupported.
    """
    if isinstance(x, jax.Array):
        return x
    if isinstance(x, (numbers.Number, np.ndarray)):
        return jnp.asarray(x)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "numpy"):
        return jnp.asarray(x.numpy())
    if isinstance(x, (Mapping, Sequence)):
        return jax.tree.map(as_jax_tensor, x)
    raise NotImplementedError(f"{type(x)}: {x}")


def as_torch_tensor(x: Any):
    """Converts `x` to torch Tensor recursively.

    Args:
        x: a jnp array, numpy array, TF/PyTorch Tensor, or a nested structure of arrays or Tensors.

    Returns:
        A nested structure with the same structure as `x` but with values converted to torch Tensors.

    Raises:
        NotImplementedError: If conversion for the input type is unsupported.
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, jax.Array):
        array = np.array(x)
        return torch.from_numpy(array)
    if isinstance(x, (numbers.Number, np.ndarray)):
        return torch.Tensor(x)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "numpy"):
        return torch.from_numpy(x.numpy())
    if isinstance(x, (Mapping, Sequence)):
        return jax.tree.map(as_torch_tensor, x)
    raise NotImplementedError(f"{type(x)}: {x}")


def transformers_rope_to_axlearn_rope(vector: torch.Tensor) -> torch.Tensor:
    """Permutes q and k vector because transformers package has a different implementation of RoPE.

    The revert operation of the following:
    https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736b10dae/src/transformers/models/llama/convert_llama_weights_to_hf.py#L136
    """
    n, h, d = vector.shape
    vector = vector.view(n, 2, h // 2, d).transpose(1, 2)
    return vector.reshape(n, h, d)


def axlearn_rope_to_transformers_rope(vector: jax.Array) -> jax.Array:
    """Reverse of transformers_rope_to_axlearn_rope function."""
    n, h, d = vector.shape
    vector = vector.view(n, h // 2, 2, d).transpose(1, 2)
    return vector.reshape(n, h, d)


def parameters_from_llama(llama: LlamaForCausalLM, state: dict, version=2) -> dict:
    """Converts llama weights from huggingface model to fuji state.

    The following model are supported and tested:
    - (fuji_model_name="fuji-1B-v3", llama_model_name="Llama-3.2-1B")
    - (fuji_model_name="fuji-3B-v3", llama_model_name="Llama-3.2-3B")
    - (fuji_model_name="fuji-8B-v3", llama_model_name="Llama-3.1-8B")
    - (fuji_model_name="fuji-70B-v3", llama_model_name="Llama-3.1-70B")

    Args:
        llama: A Llama model with type LlamaForCausalLM.
        state: The state of a fuji model.

    Returns:
        NestedTensor containing the same structure as state, but the weights are from llama.
    """
    # Copy the nested dict. No need to deep copy the data since it will be replaced.
    state = jax.tree.map(lambda x: x, state)
    if "lm_head" in state["decoder"]:
        if id(llama.model.embed_tokens.weight) == id(llama.lm_head.weight):
            raise ValueError("The embed_tokens and lm_head should not share weights.")
        state["decoder"]["lm_head"]["weight"] = llama.lm_head.weight
    elif id(llama.model.embed_tokens.weight) != id(llama.lm_head.weight):
        raise ValueError("The embed_tokens and lm_head should share weights")

    state["decoder"]["emb"]["token_emb"]["weight"] = llama.model.embed_tokens.weight
    gate_proj = []
    up_proj = []
    down_proj = []
    qkv = []
    o = []
    input_norm = []
    post_attention_norm = []
    o_shape = state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"][
        "o_proj"
    ]["weight"].shape
    i_shape = state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"][
        "i_proj"
    ]["i_proj"]["qkv_proj"][
        "weight"
    ].shape  # (n_layers, d, n, h)

    for layer in llama.model.layers:
        gate_proj.append(layer.mlp.gate_proj.weight.transpose(0, 1))
        up_proj.append(layer.mlp.up_proj.weight.transpose(0, 1))
        down_proj.append(layer.mlp.down_proj.weight.transpose(0, 1))

        # Llama3 and Llama2 70B uses GQA, but Llama2 7B does not
        if version == 3:
            vector = torch.concat(
                [
                    transformers_rope_to_axlearn_rope(
                        layer.self_attn.q_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
                    ),
                    transformers_rope_to_axlearn_rope(
                        layer.self_attn.k_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
                    ),
                    layer.self_attn.v_proj.weight.reshape(-1, i_shape[-1], i_shape[-3]),
                ],
                dim=0,
            ).permute(2, 0, 1)
        else:
            vector = torch.stack(
                [
                    transformers_rope_to_axlearn_rope(
                        layer.self_attn.q_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
                    ),
                    transformers_rope_to_axlearn_rope(
                        layer.self_attn.k_proj.weight.reshape(-1, i_shape[-1], i_shape[-3])
                    ),
                    layer.self_attn.v_proj.weight.reshape(-1, i_shape[-1], i_shape[-3]),
                ],
                dim=0,
            ).permute([0, 3, 1, 2])
        qkv.append(vector)
        o.append(layer.self_attn.o_proj.weight.reshape(-1, o_shape[-2], o_shape[-1]))
        input_norm.append(layer.input_layernorm.weight)
        post_attention_norm.append(layer.post_attention_layernorm.weight)
    state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear1_0"]["weight"] = (
        torch.stack(gate_proj)
    )
    state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear1_1"]["weight"] = (
        torch.stack(up_proj)
    )
    state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear2"]["weight"] = (
        torch.stack(down_proj)
    )
    state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"]["o_proj"][
        "weight"
    ] = torch.stack(o)
    state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"]["i_proj"][
        "i_proj"
    ]["qkv_proj"]["weight"] = torch.stack(qkv)
    state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["norm"]["scale"] = (
        torch.stack(input_norm)
    )
    state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["norm"]["scale"] = (
        torch.stack(post_attention_norm)
    )
    state["decoder"]["output_norm"]["scale"] = llama.model.norm.weight
    return as_jax_tensor(state)


def parameters_to_llama(state: dict, llama, version=2) -> dict:
    """Convert parameters from axlearn fuji to transformers llama."""
    state = jax.tree.map(lambda x: x, state)
    llama_state = llama.state_dict()
    num_layers = llama.config.num_hidden_layers
    hidden_size = llama.config.hidden_size

    if "lm_head" in state["decoder"]:
        llama_state["lm_head.weight"] = state["decoder"]["lm_head"]["weight"]

    llama_state["model.embed_tokens.weight"] = state["decoder"]["emb"]["token_emb"]["weight"]

    for idx in range(num_layers):
        # convert linear layers
        llama_state[f"model.layers.{idx}.mlp.gate_proj.weight"] = state["decoder"]["transformer"][
            "repeat"
        ]["layer"]["feed_forward"]["linear1_0"]["weight"][idx].transpose()
        llama_state[f"model.layers.{idx}.mlp.up_proj.weight"] = state["decoder"]["transformer"][
            "repeat"
        ]["layer"]["feed_forward"]["linear1_1"]["weight"][idx].transpose()
        llama_state[f"model.layers.{idx}.mlp.down_proj.weight"] = state["decoder"]["transformer"][
            "repeat"
        ]["layer"]["feed_forward"]["linear2"]["weight"][idx].transpose()

        # convert attention layers
        if version == 3:
            raise Exception("Conversion for Llama3 not implemented")
        else:
            qkv = jnp.permute_dims(
                state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"][
                    "i_proj"
                ]["i_proj"]["qkv_proj"]["weight"][idx],
                (0, 2, 3, 1),
            )
            # assert jnp.array_equal(as_jax_tensor(llama_state[f"model.layers.{idx}.self_attn.q_proj.weight"]), axlearn_rope_to_transformers_rope(torch.from_numpy(np.array(qkv[0]))).reshape(-1, hidden_size))
            llama_state[f"model.layers.{idx}.self_attn.q_proj.weight"] = (
                axlearn_rope_to_transformers_rope(torch.from_numpy(np.array(qkv[0]))).reshape(
                    -1, hidden_size
                )
            )
            llama_state[f"model.layers.{idx}.self_attn.k_proj.weight"] = (
                axlearn_rope_to_transformers_rope(torch.from_numpy(np.array(qkv[1]))).reshape(
                    -1, hidden_size
                )
            )
            llama_state[f"model.layers.{idx}.self_attn.v_proj.weight"] = qkv[2].reshape(
                -1, hidden_size
            )

        # convert attention layer projection
        llama_state[f"model.layers.{idx}.self_attn.o_proj.weight"] = state["decoder"][
            "transformer"
        ]["repeat"]["layer"]["self_attention"]["attention"]["o_proj"]["weight"][idx].reshape(
            hidden_size, -1
        )

        # convert normalization layers
        llama_state[f"model.layers.{idx}.input_layernorm.weight"] = state["decoder"]["transformer"][
            "repeat"
        ]["layer"]["self_attention"]["norm"]["scale"][idx]
        llama_state[f"model.layers.{idx}.post_attention_layernorm.weight"] = state["decoder"][
            "transformer"
        ]["repeat"]["layer"]["feed_forward"]["norm"]["scale"][idx]

    # convert normalization layer
    llama_state["model.norm.weight"] = state["decoder"]["output_norm"]["scale"]

    return as_torch_tensor(llama_state)


def validate_weights(fuji_model_name, llama_model_name, load_true_model=False):
    # Llama-2-7b-hf vs fuji-7B-v2
    trainer_config_map = c4_trainer.named_trainer_configs()
    trainer_config_fn = trainer_config_map[fuji_model_name]
    trainer_config = trainer_config_fn()
    model_config = trainer_config.model
    model_config.set(name="fuji-test-model")

    if fuji_model_name == "fuji-7B-v2":
        # llama2 7B does not share lm_head with embedding, but fuji does
        # need to disable lm_head sharing for fuji to match llama
        # model_config.decoder.set(lm_head=None)
        model_config.decoder.set(lm_head=LmHead.default_config())

    # initialize transformer model
    if load_true_model:
        # load model to a different device to avoid OOM
        llama = LlamaForCausalLM.from_pretrained(llama_model_name, local_files_only=True)
    else:
        # self-specify smaller config for easier validation
        config = AutoConfig.from_pretrained(
            f"{llama_model_name}_config.json",
            local_files_only=True,
        )
        llama = LlamaForCausalLM._from_config(config)

        # adjust num_layers to match the value in {llama_model_name}_config.json
        model_config.decoder.transformer.set(num_layers=llama.config.num_hidden_layers)

    # fuji model has different vocab size even for the same model size
    # this only allows you to convert true llama model to fuji, the reverse is not valid for true model
    model_config.decoder.set(vocab_size=llama.config.vocab_size)

    # llama.to("cuda:2")
    llama = llama.eval()

    # initialize fuji model
    fuji = model_config.instantiate(parent=None)
    prng_key = jax.random.PRNGKey(0)
    state = fuji.initialize_parameters_recursively(prng_key=prng_key)

    # conversion for llama2 and llama3 would be different
    # for example llama3 would use GQA and some of the model also share weights between lm_head and emb
    if llama_model_name in ["Llama-2-7b", "Llama-2-7b-hf"]:
        version = 2
    else:
        version = 3

    original_state_dict = deepcopy(llama.state_dict())
    state = parameters_from_llama(llama, state, version)
    llama_state_dict = parameters_to_llama(state, llama, version)

    for layer_name, layer_weights in original_state_dict.items():
        assert jnp.array_equal(layer_weights, llama_state_dict[layer_name])


def test_rope_conversion():
    tensor = torch.rand(4096, 32, 128)
    tensor2 = transformers_rope_to_axlearn_rope(tensor)
    tensor3 = axlearn_rope_to_transformers_rope(tensor2)
    print(jnp.array_equal(tensor, tensor2))
    print(jnp.array_equal(tensor, tensor3))


if __name__ == "__main__":
    validate_weights("fuji-7B-v2", "Llama-2-7b-hf")
    validate_weights("fuji-1B-v3", "Llama-3.2-1B")
