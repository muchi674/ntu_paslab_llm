import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from microbenchmarks import (
    run_tests,
    test_allreduce,
    test_attn_score,
    test_expert,
    test_p2p,
    test_qkvo,
    test_repeat_kv,
    test_router,
)

# Environment variables set by torch.distributed.launch
NODE_RANK = int(os.environ["GROUP_RANK"])
LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])

MAX_TP_SIZE = 8

def get_global_map(device):
    global_map = torch.zeros((WORLD_SIZE, 2), dtype=torch.int64, device=device)
    local_map = torch.tensor([NODE_RANK, WORLD_RANK], dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(global_map, local_map)
    return global_map


def print_msg(msg):
    if WORLD_RANK == 0:
        print(msg)


def run_microbenchmarks(model_path):
    # init processes
    device = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        "nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=device
    )

    global_map = get_global_map(device)
    node_rank = global_map[global_map[:, 1] == WORLD_RANK][0][0].item()
    n_nodes = global_map[-1][0] + 1

    # create global group
    global_group = dist.new_group(
        list(range(WORLD_SIZE)), use_local_synchronization=True
    )
    world_ranks = list(range(WORLD_SIZE))

    # create local group
    local_group = None
    node_to_ranks = {}
    for ni in range(n_nodes):
        ranks_on_node = global_map[global_map[:, 0] == ni][:, 1].tolist()
        node_to_ranks[ni] = ranks_on_node
        node_group = dist.new_group(
            ranks_on_node, backend="nccl", use_local_synchronization=True
        )
        if node_rank == ni:
            local_group = node_group

    # get model config
    model_config_path = f"/{model_path}/config.json"

    with open(model_config_path) as f:
        model_config = json.load(f)

    # start testing
    benchmark_results = {}

    ########## testing inter node communication ##########
    if n_nodes > 1:
        print_msg("=" * 10 + " inter node communication " + "=" * 10)

        # --- testing inter node all-reduce ---
        print_msg("testing inter node all-reduce...")
        result = run_tests(model_config, 1, test_allreduce, world_ranks, global_group)
        print_msg(f"result: {result}\n")
        benchmark_results["inter_allreduce"] = result

        # --- testing inter node p2p ---
        print_msg("testing inter node p2p...")

        # select leader of each node
        leaders = []
        for i in range(n_nodes):
            leaders.append(torch.min(global_map[global_map[:, 0] == i][:, 1]).item())

        # for each sender, receiver, test p2p latency
        inter_p2p_results = {}
        for i in range(n_nodes - 1):
            sender = leaders[i]
            receiver = leaders[i + 1]
            print_msg(f"testing on sender: rank {sender}, receiver: rank {receiver}...")
            result = run_tests(
                model_config, 1, test_p2p, [sender, receiver], global_group
            )
            inter_p2p_results[f"node{i}_to_node{i+1}"] = result

        print_msg(f"result: {inter_p2p_results}\n")
        benchmark_results["inter_p2p"] = inter_p2p_results

    ########## testing performance of each node ##########
    for node_rank in range(n_nodes):
        node_results = {}
        ranks_on_node = node_to_ranks[node_rank]

        ########## testing intra node communication ##########
        print_msg(
            "=" * 10 + f" intra node communication (node {node_rank}) " + "=" * 10
        )

        # --- testing intra node all-reduce ---
        print_msg("testing intra node all-reduce...")
        result = run_tests(model_config, 1, test_allreduce, ranks_on_node, local_group)
        print_msg(f"result: {result}\n")
        node_results["intra_allreduce"] = result

        # --- testing intra node p2p ---
        print_msg("testing intra node p2p...")

        # for each sender, receiver, test p2p latency
        n_ranks_on_node = len(ranks_on_node)
        intra_p2p_results = {}
        for i in range(n_ranks_on_node - 1):
            sender = ranks_on_node[i]
            receiver = ranks_on_node[i + 1]
            print_msg(f"testing on sender: rank {sender}, receiver: rank {receiver}...")

            result = run_tests(
                model_config, 1, test_p2p, [sender, receiver], local_group
            )
            print_msg(f"result: {result}\n")
            intra_p2p_results[f"rank{i}_to_rank{i+1}"] = result

        node_results["intra_p2p"] = intra_p2p_results

        ########## testing intra node computation ##########
        print_msg("=" * 10 + f" intra node computation (node {node_rank}) " + "=" * 10)

        # --- testing single device computation ---
        comp_tests = [
            ["expert_matmul", test_expert],
            ["router", test_router],
            ["qkvo", test_qkvo],
            ["repeat_kv", test_repeat_kv],
            ["attn_score", test_attn_score],
        ]
        for name, test_func in comp_tests:
            print_msg(f"testing single device {name}...")
            tp_results = {}
            tp_size = 1
            # note: number of ranks of a node should be 2^i
            while tp_size <= MAX_TP_SIZE:
                result = run_tests(
                    model_config, tp_size, test_func, [ranks_on_node[0]], local_group
                )
                tp_results[f"tp{tp_size}"] = result
                tp_size *= 2

            print_msg(f"result: {tp_results}\n")
            node_results[name] = tp_results


        benchmark_results[f"node{node_rank}"] = node_results

    if WORLD_RANK == 0:
        filename = "results/test.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, ensure_ascii=False, indent=4)

    dist.barrier()
    dist.destroy_process_group()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()
    run_microbenchmarks(args.model_path)
    # torchrun --nnodes=2 --node-rank=0 --nproc-per-node=2 --master-addr=10.10.10.1 --master-port=9091 run_microbenchmarks.py
