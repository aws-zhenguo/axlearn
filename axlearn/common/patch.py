import fcntl as F
import os
import random
import sys
import time

import jax
import numpy as np
import redis
from jax.experimental.multihost_utils import host_local_array_to_global_array
from jax.sharding import PartitionSpec as P

POD_UID = os.environ.get("POD_UID")
NUM_NODES = int(os.environ.get("NUM_NODES", 2))
ELASTIC_CACHE_URL = os.environ.get("ELASTIC_CACHE_URL")
random.seed(1234)

def _psum(x):
    return np.sum(x)


def patch_broadcast_one_to_all(in_tree, is_source=None):
    """Use CPU as backend for sync devices. Failing because XLA does not support CPU backend in multiprocess setting.

    jaxlib.xla_extension.XlaRuntimeError: INVALID_ARGUMENT: Multiprocess computations aren't implemented on the CPU backend.
    """
    if jax.process_count() == 1:
        return jax.tree.map(np.asarray, in_tree)

    if is_source is None:
        is_source = jax.process_index() == 0

    backend = "cpu"
    devices: np.ndarray = np.array(jax.devices(backend)).reshape(
        jax.process_count(backend), jax.local_device_count(backend)
    )
    global_mesh = jax.sharding.Mesh(devices, ("processes", "local_devices"))
    pspec = P("processes")

    def pre_jit(x):
        if is_source:
            inp = x
        else:
            inp = np.zeros_like(x)
        inp = np.expand_dims(inp, axis=0)
        return host_local_array_to_global_array(inp, global_mesh, pspec)

    def post_jit(x):
        return np.asarray(x.addressable_data(0))

    in_tree = jax.tree.map(pre_jit, in_tree)
    out_tree = jax.jit(_psum, out_shardings=jax.sharding.NamedSharding(global_mesh, P()))(in_tree)
    return jax.tree.map(post_jit, out_tree)


def create_counter(sync_file_path):
    f = open(sync_file_path, "w")
    F.flock(f, F.LOCK_EX)
    f.write("0")
    f.flush()
    F.flock(f, F.LOCK_UN)


def update_counter(sync_file_path):
    f = open(sync_file_path, "r+")
    F.flock(f, F.LOCK_EX)
    content = f.read()

    counter = int(content) + 1
    f.seek(0)
    f.write(str(counter))
    f.flush()
    F.flock(f, F.LOCK_UN)


def read_counter(sync_file_path):
    f = open(sync_file_path, "r")
    F.flock(f, F.LOCK_EX)
    content = f.read()
    F.flock(f, F.LOCK_UN)
    return content


def patch_broadcast_one_to_all_with_fs(in_tree, is_source=None):
    """Use shared fsx to sync global devices. working with 8 nodes for a few steps, but failing with more steps and more nodes."""
    if is_source is None:
        is_source = jax.process_index() == 0

    sync_file_path = f"/shared/jax/{in_tree}.npy"
    if is_source:
        create_counter(sync_file_path)
    else:
        time.sleep(1)

    while True:
        try:
            print("trying to update file")
            if os.path.exists(sync_file_path):
                time.sleep(0.5)
                update_counter(sync_file_path)
                break
        except OSError as e:
            print("retry to update counter")
            time.sleep(random.random())
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    while True:
        try:
            content = read_counter(sync_file_path)

            print("file content", content)
            counter = int(content)
            if counter == NUM_NODES:
                return in_tree
            time.sleep(random.random())
        except OSError as e:
            print("retry to read counter")
            time.sleep(random.random())
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise


def patch_broadcast_one_to_all_with_redis(in_tree, is_source=None):
    r = redis.from_url(
        ELASTIC_CACHE_URL,
        health_check_interval=10,
        socket_connect_timeout=5,
        retry_on_timeout=True,
        socket_keepalive=True,
    )
    key = f"{POD_UID}:{in_tree}:{random.random()}"

    r.incr(key)
    while True:
        counter = int(r.get(key).decode())
        if counter == NUM_NODES:
            r.expire(key, 60 * 30)
            return in_tree
        elif counter > NUM_NODES:
            raise Exception(f"duplicate sync key used! {key} -> {counter}")


def patch_all():
    print("applying simulation patch...")
    jax.experimental.multihost_utils.broadcast_one_to_all = patch_broadcast_one_to_all_with_redis
