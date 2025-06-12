from operator import itemgetter
from pathlib import Path
import argparse
import json
import logging
import random
import torch

MODEL_SPECS = {
    "precision_bytes": 2,
    "n_layers": 32,
    "model_d": 4096,
    "vocab_d": 32000,
    # below are PER LAYER, PER TOKEN statistics
    # model_d = 4096
    # n_heads = 32
    # n_kv_heads = 8
    # head_dim = model_d // n_heads = 128
    # wq.shape = (model_d, n_heads * head_dim)
    # wk.shape = (model_d, n_kv_heads * head_dim)
    # wv.shape = (model_d, n_kv_heads * head_dim)
    # wo.shape = (n_heads * head_dim, model_d)
    # n_params = 2 * model_d * head_dim * (n_heads + n_kv_heads)
    # FLOPS of matmul between 2 matrices (m, k), (k, n) = 2 * m * n * k
    # since we are calculating per token stats, m = 1, thus FLOPS
    # essentially equals 2 * n * k, which is 2 * n_params
    "attn": {
        "param_bytes": 2 * 4096 * 128 * (32 + 8) * 2,
        "flops": 2 * 2 * 4096 * 128 * (32 + 8),
    },
    "expert": {
        "n_experts": 8,
        "top_k": 2,
        # stat below are for ONE EXPERT
        "param_bytes": 3 * 4096 * 14336 * 2,
        "flops": 3 * 2 * 4096 * 14336,
    },
}

SETUP = [
    {
        "gpu_id": "4090",
        "n_gpus": 2,
    },
    {
        "gpu_id": "4090",
        "n_gpus": 4,
    },
    # {
    #     "gpu_id": "4090",
    #     "n_gpus": 2,
    # },
]


def get_json(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def distribute(n_items, n_bins):
    # from: https://stackoverflow.com/questions/54353083/distribute-an-integer-amount-by-a-set-of-slots-as-evenly-as-possible
    base, extra = divmod(n_items, n_bins)
    return [base + (i < extra) for i in range(n_bins)]


def find_parallel_strategies(batch_size: int, prompt_len: int):
    strategies = {}

    total_n_gpus = sum(node["n_gpus"] for node in SETUP)
    pp_gpu_layers = distribute(MODEL_SPECS["n_layers"], total_n_gpus)
    ep_gpu_experts = distribute(MODEL_SPECS["expert"]["n_experts"], total_n_gpus)
    pp_node_layers = []
    ep_node_experts = []
    i = 0
    for node in SETUP:
        j = i + node["n_gpus"]
        pp_node_layers.append(sum(pp_gpu_layers[i:j]))
        ep_node_experts.append(sum(ep_gpu_experts[i:j]))
        i = j

    # strategies["naive PP"] = {
    #     "batch_size": batch_size,
    #     "prompt_len": prompt_len,
    #     "pp_strategy": {"is_naive": True, "pp_node_layers": pp_node_layers},
    # }
    # strategies["inter-attn-inter-experts TP"] = {
    #     "batch_size": batch_size,
    #     "prompt_len": prompt_len,
    #     "attn_strategy": {"attn_is_intra": False, "attn_parallelism": "tp"},
    #     "experts_strategy": {"experts_are_intra": False, "experts_parallelism": "tp"},
    # }
    # strategies["intra-attn-inter-experts TP"] = {
    #     "batch_size": batch_size,
    #     "prompt_len": prompt_len,
    #     "attn_strategy": {"attn_is_intra": True, "attn_parallelism": "tp"},
    #     "experts_strategy": {"experts_are_intra": False, "experts_parallelism": "tp"},
    # }
    # strategies["inter-experts TP"] = {
    #     "batch_size": batch_size,
    #     "prompt_len": prompt_len,
    #     "experts_strategy": {"experts_are_intra": False, "experts_parallelism": "tp"},
    # }
    # strategies["inter EP"] = {
    #     "batch_size": batch_size,
    #     "prompt_len": prompt_len,
    #     "experts_strategy": {
    #         "experts_are_intra": False,
    #         "experts_parallelism": "ep",
    #         "experts_allocation": [3, 5],
    #     },
    # }
    # strategies["inter PP + intra-experts TP"] = {
    #     "batch_size": batch_size,
    #     "prompt_len": prompt_len,
    #     "pp_strategy": {"is_naive": False, "pp_node_layers": pp_node_layers},
    #     "experts_strategy": {"experts_are_intra": True, "experts_parallelism": "tp"},
    # }
    # strategies["inter PP + intra-attn-intra-experts TP"] = {
    #     "batch_size": batch_size,
    #     "prompt_len": prompt_len,
    #     "pp_strategy": {"is_naive": False, "pp_node_layers": pp_node_layers},
    #     "attn_strategy": {"attn_is_intra": True, "attn_parallelism": "tp"},
    #     "experts_strategy": {"experts_are_intra": True, "experts_parallelism": "tp"},
    # }
    # strategies["inter PP + intra EP"] = {
    #     "batch_size": batch_size,
    #     "prompt_len": prompt_len,
    #     "pp_strategy": {"is_naive": False, "pp_node_layers": pp_node_layers},
    #     "experts_strategy": {"experts_are_intra": True, "experts_parallelism": "ep"},
    # }
    strategies["inter PP + intra EP + intra-attn TP"] = {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "pp_strategy": {"is_naive": False, "pp_node_layers": pp_node_layers},
        "attn_strategy": {"attn_is_intra": True, "attn_parallelism": "tp"},
        "experts_strategy": {"experts_are_intra": True, "experts_parallelism": "ep"},
    }
    # strategies["inter EP + intra-experts TP"] = {
    #     "batch_size": batch_size,
    #     "prompt_len": prompt_len,
    #     "experts_strategy": {
    #         "experts_are_intra": False,
    #         "experts_parallelism": "ep+tp",
    #         "experts_allocation": [3, 5],
    #     },
    # }
    # strategies["inter EP + intra-attn-intra-experts TP"] = {
    #     "batch_size": batch_size,
    #     "prompt_len": prompt_len,
    #     "attn_strategy": {"attn_is_intra": True, "attn_parallelism": "tp"},
    #     "experts_strategy": {
    #         "experts_are_intra": False,
    #         "experts_parallelism": "ep+tp",
    #         "experts_allocation": [3, 5],
    #     },
    # }
    # strategies["inter EP + intra-attn TP"] = {
    #     "batch_size": batch_size,
    #     "prompt_len": prompt_len,
    #     "attn_strategy": {"attn_is_intra": True, "attn_parallelism": "tp"},
    #     "experts_strategy": {
    #         "experts_are_intra": False,
    #         "experts_parallelism": "ep",
    #         "experts_allocation": [3, 5],
    #     },
    # }
    # strategies["inter EP + inter-attn-intra-experts TP"] = {
    #     "batch_size": batch_size,
    #     "prompt_len": prompt_len,
    #     "attn_strategy": {"attn_is_intra": False, "attn_parallelism": "tp"},
    #     "experts_strategy": {
    #         "experts_are_intra": False,
    #         "experts_parallelism": "ep+tp",
    #         "experts_allocation": [3, 5],
    #     },
    # }

    return strategies


def ceildiv(a, b):
    # from: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)


def estimate_lower_bound_exec_time(
    bench_res: dict,
    batch_size: int,
    prompt_len: int,
    pp_strategy: dict = {},
    attn_strategy: dict = {},
    experts_strategy: dict = {},
):
    # TODO:
    # 1. we are yet to adjust compute time for extra long sequences, which requires
    # substantially more data to be moved from memory to cache and more FLOPs
    # 2. we are yet to account for data movement cost for KV-cache
    # OBSERVATION:
    # communication data size is hugely dependent on implementation
    precision_bytes, n_layers, model_d, vocab_d, attn_specs, expert_specs = itemgetter(
        "precision_bytes", "n_layers", "model_d", "vocab_d", "attn", "expert"
    )(MODEL_SPECS)
    total_n_gpus = sum(node["n_gpus"] for node in SETUP)

    pp_is_naive = pp_strategy.get("is_naive")
    pp_node_layers = pp_strategy.get(
        "pp_node_layers"
    )  # should follow the same ordering as SETUP
    attn_is_intra = attn_strategy.get("attn_is_intra")
    attn_parallelism = attn_strategy.get("attn_parallelism")
    experts_are_intra = experts_strategy.get("experts_are_intra")
    experts_parallelism = experts_strategy.get("experts_parallelism")
    experts_allocation = experts_strategy.get("experts_allocation")
    n_experts = expert_specs["n_experts"]
    top_k = expert_specs["top_k"]

    input_shape = f"{batch_size}-{prompt_len}"
    exec_time_by_node = []
    for node_idx, node in enumerate(SETUP):
        bench_res_node = bench_res[f"node{node_idx}"]
        exec_time = []
        n_local_gpus = node["n_gpus"]

        if attn_parallelism is None:
            parallel_size = 1
            comm_time = 0

        elif attn_parallelism == "tp":  # partitions weights
            # or attn_parallelism == "dp"  # partitions input
            # or attn_parallelism == "cp"  # partitions input
            parallel_size = n_local_gpus if attn_is_intra else total_n_gpus
            if attn_is_intra:
                comm_time = bench_res_node["intra_allreduce"][input_shape]
            else:
                comm_time = bench_res["inter_allreduce"][input_shape]

        compute_time = (
            bench_res_node["qkvo"][f"tp{parallel_size}"][input_shape]
            + bench_res_node["repeat_kv"][f"tp{parallel_size}"][input_shape]
            + bench_res_node["attn_score"][f"tp{parallel_size}"][input_shape]
        )
        exec_time.extend([compute_time, comm_time])

        # TODO: for now, we are assuming that expert selection follows an uniform dist
        # expert selection sampling
        n_sampling = 100
        n_tokens = batch_size * prompt_len
        n_activations = []
        for _ in range(n_sampling):
            n_activation = [0 for _ in range(n_experts)]
            for i in range(n_tokens):
                activated_experts = random.sample(range(n_experts), top_k)
                for e in activated_experts:
                    n_activation[e] += 1
            
            n_activations.append(n_activation)

        intra_node_comm_time = bench_res_node["intra_allreduce"][input_shape]
        inter_node_comm_time = bench_res["inter_allreduce"][input_shape]
        if experts_parallelism is None:
            parallel_size = 1
            n_local_experts = n_experts
            comm_time = 0

        elif experts_parallelism == "tp":
            parallel_size = n_local_gpus if experts_are_intra else total_n_gpus
            n_local_experts = n_experts
            comm_time = (
                intra_node_comm_time if experts_are_intra else inter_node_comm_time
            )

        elif experts_parallelism == "ep":
            parallel_size = 1
            # NOTE: consider maximum number of local expert
            n_local_experts = (
                n_local_gpus
                - 1
                + (n_experts if experts_are_intra else experts_allocation[node_idx])
            ) // n_local_gpus
            comm_time = (
                intra_node_comm_time if experts_are_intra else inter_node_comm_time
            )

        elif experts_parallelism == "ep+tp":
            assert not experts_are_intra
            parallel_size = n_local_gpus
            n_local_experts = experts_allocation[node_idx]
            comm_time = inter_node_comm_time
        
        total_comp_time = 0
        for n_activation in n_activations:
            for ei in range(n_local_experts):
                if n_activation[ei] > 0:
                    total_comp_time += bench_res_node["expert_matmul"][f"tp{parallel_size}"][str(n_activation[ei])]
            
        compute_time = total_comp_time / len(n_activations) + bench_res_node["router"]["tp1"][input_shape]
        exec_time.extend([compute_time, comm_time])

        constant = pp_node_layers[node_idx] if pp_strategy else n_layers
        exec_time = [val * constant for val in exec_time]

        extra_comm_time = 0.0
        if pp_is_naive:
            for i in range(node["n_gpus"]-1):
                extra_comm_time += bench_res_node["intra_p2p"][f"rank{i}_to_rank{i+1}"][input_shape]

        if pp_strategy:
            if node_idx < len(SETUP) - 1:
                extra_comm_time += bench_res["inter_p2p"][f"node{node_idx}_to_node{node_idx+1}"][input_shape]
            else:
                # last node broadcasts output to other nodes
                c = ceildiv(vocab_d, model_d)
                k = f"{batch_size}-{1}"
                extra_comm_time += bench_res["inter_p2p"]["node0_to_node1"][k] * c * (node["n_gpus"]-1)
        
        exec_time.append(extra_comm_time)
        exec_time_by_node.append(exec_time)

    exec_time_by_node = torch.tensor(exec_time_by_node)  # in seconds
    breakdown: torch.Tensor
    if pp_strategy:
        breakdown = torch.sum(exec_time_by_node, dim=0)
    else:
        breakdown = torch.max(exec_time_by_node, dim=0)[0]
    total_exec_time = torch.sum(breakdown).item()
    throughput = batch_size * prompt_len * 1000 / total_exec_time

    # breakdown in ms, throughput in t/s
    return breakdown.tolist(), total_exec_time, throughput


def run_perf_model(bench_res: dict, batch_size: int, prompt_len: int, sort: bool):
    res = []
    strategies = find_parallel_strategies(batch_size, prompt_len)
    for name, args in strategies.items():
        breakdown, total_exec_time, throughput = estimate_lower_bound_exec_time(
            bench_res=bench_res, **args
        )
        res.append(
            [name, batch_size, prompt_len] + breakdown + [total_exec_time, throughput]
        )

    if sort:
        sorted_indices = torch.argsort(
            torch.tensor([row[-1] for row in res]), descending=True
        )
        res = [res[i] for i in sorted_indices]

    for row in res:
        print(
            ", ".join(
                [val if isinstance(val, str) else str(round(val, 3)) for val in row]
            )
        )


def main(
    bench_res: str,
    start_batch_size: int,
    start_prompt_len: int,
    end_batch_size: int = None,
    end_prompt_len: int = None,
    sort: bool = False,
):
    bench_res = get_json(Path(bench_res))
    cols = [
        "strategy",
        "batch_size",
        "prompt_len",
        "attn_compute",
        "attn_comm",
        "experts_compute",
        "experts_comm",
        "extra_comm",
        "total",
        "t/s",
    ]
    print(", ".join(cols))
    end_batch_size = end_batch_size or start_batch_size
    end_prompt_len = end_prompt_len or start_prompt_len

    # print decode results
    bs = start_batch_size
    while bs <= end_batch_size:
        run_perf_model(bench_res, bs, 1, sort)
        bs *= 2

    # prefill
    while start_batch_size <= end_batch_size:
        p_len = start_prompt_len
        while p_len <= end_prompt_len:
            run_perf_model(bench_res, start_batch_size, p_len, sort)
            p_len *= 2
        start_batch_size *= 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-res", type=str)
    parser.add_argument("--start-bs", type=int)
    parser.add_argument("--end-bs", type=int)
    parser.add_argument("--start-plen", type=int)
    parser.add_argument("--end-plen", type=int)
    parser.add_argument("--sort", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(
        args.bench_res,
        args.start_bs,
        args.start_plen,
        args.end_bs,
        args.end_plen,
        args.sort,
    )
