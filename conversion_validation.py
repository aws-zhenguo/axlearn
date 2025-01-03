import numpy as np

DEFAULT_K = 64
DEFAULT_P = 0.99


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
                jaccard_index = len(top_x_set.intersection(top_y_set)) / len(
                    top_x_set.union(top_y_set)
                )
                jaccard_indices[k] += jaccard_index

    for k in jaccard_indices:
        jaccard_indices[k] /= batch_size * seq_len
    return jaccard_indices


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


def get_top_k_smallest(x, y, k=DEFAULT_K):
    x_indices = np.argsort(x, axis=-1)[:, :, ::-1]
    top_x = ndarray_apply_order(x, x_indices)[:, :, :k]
    top_y = ndarray_apply_order(y, x_indices)[:, :, :k]
    return min(np.min(top_x), np.min(top_y))


def assert_top_p_allclose(x, y, atol, rtol, p=DEFAULT_P):
    # x = np.sort(x, axis=-1)[:,:,::-1]
    # y = np.sort(y, axis=-1)[:,:,::-1]
    x_indices = np.argsort(x, axis=-1)[:, :, ::-1]

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
    print("smallest prob in top p:", min(np.min(top_x_flat), np.min(top_y_flat)))

    rdiff = relative_difference(top_x_flat, top_y_flat)
    print("max top p difference", np.max(np.abs(top_x_flat - top_y_flat)))
    print("mean top p difference", np.mean(np.abs(top_x_flat - top_y_flat)))
    print("max top p relative difference", np.max(rdiff))
    print("mean top p relative difference", np.mean(rdiff))

    np.testing.assert_allclose(top_x_flat, top_y_flat, atol=atol, rtol=rtol, equal_nan=False)


def assert_top_k_allclose(x, y, atol, rtol, k=DEFAULT_K):
    x_indices = np.argsort(x, axis=-1)[:, :, ::-1]
    top_x = ndarray_apply_order(x, x_indices)[:, :, :k]
    top_y = ndarray_apply_order(y, x_indices)[:, :, :k]
    # x = np.sort(x, axis=-1)[:,:,::-1]
    # y = np.sort(y, axis=-1)[:,:,::-1]
    print("smallest prob in top k:", min(np.min(top_x), np.min(top_y)))

    rdiff = relative_difference(top_x, top_y)
    print("max top k difference", np.max(np.abs(top_x - top_y)))
    print("mean top k difference", np.mean(np.abs(top_x - top_y)))
    print("max top k relative difference", np.max(rdiff))
    print("mean top k relative difference", np.mean(rdiff))

    np.testing.assert_allclose(top_x, top_y, atol=atol, rtol=rtol, equal_nan=False)


def assert_top_p_and_top_k_allclose(
    x, y, atol_high, rtol_high, atol_low, rtol_low, threshold=1e-3, k=DEFAULT_K, p=DEFAULT_P
):
    """
    the probability distribution has a long tail therefore it is not feasible
    to have good single atol and rtol to both pass large values and cover small
    values at the same time. So breaking down values into high and low values
    """
    x_indices = np.argsort(x, axis=-1)[:, :, ::-1]
    y_indices = np.argsort(y, axis=-1)[:, :, ::-1]

    top_x_flat = list()
    top_y_flat = list()
    for i, (x_seq, y_seq) in enumerate(zip(x, y)):
        for j, (x_token, y_token) in enumerate(zip(x_seq, y_seq)):
            top_x_indices = list()
            x_total = 0.0

            # top p / k in x
            for idx in x_indices[i][j]:
                x_total += x_token[idx]
                top_x_indices.append(idx)

                if x_total > p and len(top_x_indices) >= k:
                    break

            top_y_indices = list()
            y_total = 0.0
            # top p / k in y
            for idx in y_indices[i][j]:
                y_total += y_token[idx]
                top_y_indices.append(idx)

                if y_total > p and len(top_y_indices) >= k:
                    break

            top_indices = list(set(top_x_indices).union(set(top_y_indices)))
            top_x = x_token[top_indices]
            top_y = y_token[top_indices]

            top_x_flat.extend(top_x)
            top_y_flat.extend(top_y)

    top_x_flat = np.asarray(top_x_flat)
    top_y_flat = np.asarray(top_y_flat)

    high_indices = np.where(top_x_flat > threshold)
    top_x_flat_high = top_x_flat[high_indices]
    top_y_flat_high = top_y_flat[high_indices]
    low_indices = np.where(top_x_flat <= threshold)
    top_x_flat_low = top_x_flat[low_indices]
    top_y_flat_low = top_y_flat[low_indices]

    print("checkping higher end:")
    print(
        "smallest prob in top k and top p:", min(np.min(top_x_flat_high), np.min(top_y_flat_high))
    )

    rdiff = relative_difference(top_x_flat_high, top_y_flat_high)
    print("max top p/k difference", np.max(np.abs(top_x_flat_high - top_y_flat_high)))
    print("mean top p/k difference", np.mean(np.abs(top_x_flat_high - top_y_flat_high)))
    print("max top p/k relative difference", np.max(rdiff))
    print("mean top p/k relative difference", np.mean(rdiff))

    np.testing.assert_allclose(
        top_x_flat_high, top_y_flat_high, atol=atol_high, rtol=rtol_high, equal_nan=False
    )

    print("checkping lower end:")
    print("smallest prob in top k and top p:", min(np.min(top_x_flat_low), np.min(top_y_flat_low)))

    rdiff = relative_difference(top_x_flat_low, top_y_flat_low)
    print("max top p/k difference", np.max(np.abs(top_x_flat_low - top_y_flat_low)))
    print("mean top p/k difference", np.mean(np.abs(top_x_flat_low - top_y_flat_low)))
    print("max top p/k relative difference", np.max(rdiff))
    print("mean top p/k relative difference", np.mean(rdiff))

    np.testing.assert_allclose(
        top_x_flat_low, top_y_flat_low, atol=atol_low, rtol=rtol_low, equal_nan=False
    )


def validate_probs(fuji_model_name, llama_model_name):
    fuji_probs = np.load(f"{fuji_model_name}_probs.npy")
    llama_probs = np.load(f"{llama_model_name}_probs.npy")

    # The difference is caused by the SDPA attention layer. The deeper the larger the error.
    atol_high = 3.5e-5
    rtol_high = 7e-3
    atol_low = 1.5e-6
    rtol_low = 1e-3

    top_k_smallest = get_top_k_smallest(llama_probs, fuji_probs)
    # Selecting a threshold per discussion with Hah
    threshold_min, threshold_max = 1e-5, 1e-4
    threshold = min(top_k_smallest, threshold_max)
    threshold = max(threshold, threshold_min)
    print("threshold:", threshold)

    # (0.00065897405 - 3.4e-5) / 0.19473135
    # assert_top_k_allclose(llama_probs, fuji_probs, atol, rtol)
    # assert_top_p_allclose(llama_probs, fuji_probs, atol, rtol, p=0.99)
    assert_top_p_and_top_k_allclose(
        llama_probs,
        fuji_probs,
        atol_high,
        rtol_high,
        atol_low,
        rtol_low,
        threshold=threshold,
        p=0.99,
    )


texts = [
    "How are you doing?",
    "who is the president of the US now?",
    "The USA is in which continent?",
    "California is a state in",
    "Can you tell me something about California state?\n",
]


def run_gpu_checkpoint_tests(load_true_model=False):
    from axlearn_inference import validate_conversion

    print("Converting and validating Axlearn GPU to HF on 7B...")
    validate_conversion(
        "fuji-7B-v2",
        "Llama-2-7b-hf",
        load_true_model=load_true_model,
        reverse=True,
        texts=texts,
        fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/axlearn_venv/baselines/10976/axlearn_out/checkpoints/step_00034000",
        trn_checkpoint=False,
    )
    validate_probs("fuji-7B-v2", "Llama-2-7b-hf")

    print("Converting and validating Axlearn GPU to HF on 70B...")
    validate_conversion(
        "fuji-70B-v2",
        "Llama-2-70b-hf",
        load_true_model=load_true_model,
        reverse=True,
        texts=texts,
        use_gqa=True,
        trn_checkpoint=False,
    )
    validate_probs("fuji-70B-v2", "Llama-2-70b-hf")


def run_trn_checkpoint_tests(load_true_model=False):
    from axlearn_inference import validate_conversion

    print(
        "Make sure StackedTransformerLayer and GroupedQKVLinear are used when converting TRN ckpt!"
    )
    print("Converting and validating Axlearn TRN to HF on 7B...")
    validate_conversion(
        "fuji-7B-v2",
        "Llama-2-7b-hf",
        load_true_model=load_true_model,
        reverse=True,
        texts=texts,
        trn_checkpoint=True,
    )
    validate_probs("fuji-7B-v2", "Llama-2-7b-hf")

    print("Converting and validating Axlearn TRN to HF on 70B...")
    validate_conversion(
        "fuji-70B-v2",
        "Llama-2-70b-hf",
        load_true_model=load_true_model,
        reverse=True,
        texts=texts,
        fuji_model_path="/fsx/czhenguo/Projects/fruitstand/runs/artifacts/241230232345/axlearn_out/checkpoints/step_00000002",
        use_gqa=True,
        trn_checkpoint=True,
    )
    validate_probs("fuji-70B-v2", "Llama-2-70b-hf")

    # TODO add HF to Axlearn test case


if __name__ == "__main__":
    # validate_probs("fuji-7B-v2", "Llama-2-7b-hf")
    # validate_probs("fuji-70B-v2", "Llama-2-70b-hf")
    # run_gpu_checkpoint_tests()
    run_trn_checkpoint_tests()
