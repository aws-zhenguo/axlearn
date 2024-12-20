import numbers
from collections.abc import Mapping, Sequence
from typing import Any, Callable, NamedTuple, Optional, TypeVar, Union

import jax
import numpy as np
import torch
from jax import numpy as jnp
from copy import deepcopy


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
    """Reverse of transformers_rope_to_axlearn_rope function.
    """
    n, h, d = vector.shape
    vector = vector.view(n, h // 2, 2, d).transpose(1, 2)
    return vector.reshape(n, h, d)


def parameters_to_llama(state: dict, llama, version=2) -> dict:
    state = jax.tree.map(lambda x: x, state)
    llama_state = llama.state_dict()
    # TODO remove
    llama_state_copy = deepcopy(llama_state)
    num_layers = llama.config.num_hidden_layers
    hidden_size = llama.config.hidden_size

    if "lm_head" in state["decoder"]:
        llama_state["lm_head.weight"] = state["decoder"]["lm_head"]["weight"]

    print(jnp.array_equal(llama_state["model.embed_tokens.weight"], state["decoder"]["emb"]["token_emb"]["weight"]))
    llama_state["model.embed_tokens.weight"] = state["decoder"]["emb"]["token_emb"]["weight"]

    for idx in range(num_layers):
        # convert linear layers
        print(jnp.array_equal(as_jax_tensor(llama_state[f"model.layers.{idx}.mlp.gate_proj.weight"]), state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear1_0"]["weight"][idx].transpose()))
        llama_state[f"model.layers.{idx}.mlp.gate_proj.weight"] = state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear1_0"]["weight"][idx].transpose()
        print(jnp.array_equal(as_jax_tensor(llama_state[f"model.layers.{idx}.mlp.up_proj.weight"]), state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear1_1"]["weight"][idx].transpose()))
        llama_state[f"model.layers.{idx}.mlp.up_proj.weight"] = state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear1_1"]["weight"][idx].transpose()
        print(jnp.array_equal(as_jax_tensor(llama_state[f"model.layers.{idx}.mlp.down_proj.weight"]), state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear2"]["weight"][idx].transpose()))
        llama_state[f"model.layers.{idx}.mlp.down_proj.weight"] = state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["linear2"]["weight"][idx].transpose()

        # convert attention layers
        if version == 3:
            raise Exception("Conversion for Llama3 not implemented")
        else:
            qkv = jnp.permute_dims(state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"]["i_proj"]["i_proj"]["qkv_proj"]["weight"][idx], (0, 2, 3, 1))
            print(jnp.array_equal(as_jax_tensor(llama_state[f"model.layers.{idx}.self_attn.q_proj.weight"]), axlearn_rope_to_transformers_rope(torch.from_numpy(np.array(qkv[0]))).reshape(-1, hidden_size)))
            llama_state[f"model.layers.{idx}.self_attn.q_proj.weight"] = axlearn_rope_to_transformers_rope(torch.from_numpy(np.array(qkv[0]))).reshape(-1, hidden_size)
            print(jnp.array_equal(as_jax_tensor(llama_state[f"model.layers.{idx}.self_attn.k_proj.weight"]), axlearn_rope_to_transformers_rope(torch.from_numpy(np.array(qkv[1]))).reshape(-1, hidden_size)))
            llama_state[f"model.layers.{idx}.self_attn.k_proj.weight"] = axlearn_rope_to_transformers_rope(torch.from_numpy(np.array(qkv[1]))).reshape(-1, hidden_size)
            print(jnp.array_equal(as_jax_tensor(llama_state[f"model.layers.{idx}.self_attn.v_proj.weight"]), qkv[2].reshape(-1, hidden_size)))
            llama_state[f"model.layers.{idx}.self_attn.v_proj.weight"] = qkv[2].reshape(-1, hidden_size)

        # convert attention layer projection
        print(jnp.array_equal(as_jax_tensor(llama_state[f"model.layers.{idx}.self_attn.o_proj.weight"]), state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"]["o_proj"]["weight"][idx].reshape(hidden_size, -1)))
        llama_state[f"model.layers.{idx}.self_attn.o_proj.weight"] = state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["attention"]["o_proj"]["weight"][idx].reshape(hidden_size, -1)

        # convert normalization layers
        print(jnp.array_equal(as_jax_tensor(llama_state[f"model.layers.{idx}.input_layernorm.weight"]), state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["norm"]["scale"][idx]))
        llama_state[f"model.layers.{idx}.input_layernorm.weight"] = state["decoder"]["transformer"]["repeat"]["layer"]["self_attention"]["norm"]["scale"][idx]
        print(jnp.array_equal(as_jax_tensor(llama_state[f"model.layers.{idx}.post_attention_layernorm.weight"]), state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["norm"]["scale"][idx]))
        llama_state[f"model.layers.{idx}.post_attention_layernorm.weight"] = state["decoder"]["transformer"]["repeat"]["layer"]["feed_forward"]["norm"]["scale"][idx]

    # convert normalization layer
    llama_state["model.norm.weight"] = state["decoder"]["output_norm"]["scale"]

    return llama_state

if __name__ == "__main__":
    tensor = torch.rand(4096, 32, 128)
    tensor2 = transformers_rope_to_axlearn_rope(tensor)
    tensor3 = axlearn_rope_to_transformers_rope(tensor2)
    print(jnp.array_equal(tensor, tensor2))
    print(jnp.array_equal(tensor, tensor3))
