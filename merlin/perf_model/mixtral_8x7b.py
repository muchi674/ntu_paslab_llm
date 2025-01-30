from operator import itemgetter

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
HW_SPECS = {
    # TODO: write micro-benchmark programs to test this
    "inter_comm": {
        "collective_overhead": 150 / 1000**2,  # in microsecond
        "collective_bandwidth": 1.25 * (10**9),
        "p2p_overhead": 50 / 1000**2,
        "p2p_bandwidth": 1.25 * (10**9),
    },
    "4090": {
        "bf16_flops": 165.2 * (10**12),
        "mem_bw": 1008 * (10**9),
    },
}
SETUP = [
    {
        # TODO: write micro-benchmark programs to test this
        "intra_comm": {
            "collective_overhead": 20 / 1000**2,  # in microsecond
            "collective_bandwidth": 20 * (10**9),
            "p2p_overhead": 10 / 1000**2,
            "p2p_bandwidth": 30 * (10**9),
        },
        "gpu_id": "4090",
        "n_gpus": 2,
    },
    {
        "intra_comm": {
            "collective_overhead": 20 / 1000**2,  # in microsecond
            "collective_bandwidth": 20 * (10**9),
            "p2p_overhead": 10 / 1000**2,
            "p2p_bandwidth": 30 * (10**9),
        },
        "gpu_id": "4090",
        "n_gpus": 4,
    },
    {
        "intra_comm": {
            "collective_overhead": 20 / 1000**2,  # in microsecond
            "collective_bandwidth": 20 * (10**9),
            "p2p_overhead": 10 / 1000**2,
            "p2p_bandwidth": 30 * (10**9),
        },
        "gpu_id": "4090",
        "n_gpus": 2,
    },
]


def ceildiv(a, b):
    # from: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)

def find_parallel_strategies():
    # TODO: given SETUP, output optimal cut for
    # PP, TP, EP, PP+TP, PP+EP, TP+EP
    # we can leave DP and CP for now
    pass

def estimate_lower_bound_exec_time(
    batch_size: int,
    prompt_len: int,
    pp_strategy: dict,
    attn_strategy: dict,
    experts_strategy: dict,
):
    precision_bytes, n_layers, model_d, vocab_d, attn_specs, expert_specs = itemgetter(
        "precision_bytes", "n_layers", "model_d", "vocab_d", "attn", "expert"
    )(MODEL_SPECS)
    inter_coll_oh, inter_coll_bw, inter_p2p_oh, inter_p2p_bw = itemgetter(
        "collective_overhead", "collective_bandwidth", "p2p_overhead", "p2p_bandwidth"
    )(HW_SPECS["inter_comm"])
    total_n_gpus = sum(node["n_gpus"] for node in SETUP)

    pp_is_naive = pp_strategy.get("is_naive")
    pp_node_layers = pp_strategy.get(
        "pp_node_layers"
    )  # should follow the same ordering as SETUP
    attn_is_intra = attn_strategy.get("attn_is_intra")
    attn_parallelism = attn_strategy.get("attn_parallelism")
    attn_param_bytes = attn_specs["param_bytes"]
    attn_flops = attn_specs["flops"] * batch_size * prompt_len
    experts_are_intra = experts_strategy.get("experts_are_intra")
    experts_parallelism = experts_strategy.get("experts_parallelism")
    experts_allocation = experts_strategy.get("experts_allocation")
    expert_param_bytes = expert_specs["param_bytes"]
    expert_flops = expert_specs["flops"]
    n_experts = expert_specs["n_experts"]
    top_k = expert_specs["top_k"]

    comm_data_size = precision_bytes * batch_size * prompt_len * model_d
    exec_time_by_node = []
    for node_idx, node in enumerate(SETUP):
        exec_time = []
        intra_coll_oh, intra_coll_bw, intra_p2p_oh, intra_p2p_bw = itemgetter(
            "collective_overhead",
            "collective_bandwidth",
            "p2p_overhead",
            "p2p_bandwidth",
        )(node["intra_comm"])
        n_local_gpus = node["n_gpus"]
        gpu_mem_bw = HW_SPECS[node["gpu_id"]]["mem_bw"]
        gpu_flops = HW_SPECS[node["gpu_id"]]["bf16_flops"]

        if attn_parallelism is None:
            compute_time = max(attn_param_bytes / gpu_mem_bw, attn_flops / gpu_flops)
            exec_time.extend([compute_time, 0.0])
        elif (
            attn_parallelism == "tp"  # partitions weights
            or attn_parallelism == "dp"  # partitions input
            or attn_parallelism == "cp"  # partitions input
        ):
            parallel_size = n_local_gpus if attn_is_intra else total_n_gpus
            compute_time = max(
                attn_param_bytes / parallel_size / gpu_mem_bw,
                attn_flops / parallel_size / gpu_flops,
            )
            if attn_is_intra:
                comm_time = intra_coll_oh + comm_data_size / intra_coll_bw
            else:
                comm_time = inter_coll_oh + comm_data_size / inter_coll_bw
            exec_time.extend([compute_time, comm_time])

        # TODO: for now, we are assuming that expert selection follows an uniform dist
        n_act_experts = min(batch_size * prompt_len * top_k, n_experts)
        intra_node_comm_time = intra_coll_oh + comm_data_size * top_k / intra_coll_bw
        inter_node_comm_time = inter_coll_oh + comm_data_size * top_k / inter_coll_bw
        if experts_parallelism is None:
            compute_time = max(
                n_act_experts * expert_param_bytes / gpu_mem_bw,
                batch_size * prompt_len * top_k * expert_flops / gpu_flops,
            )
            exec_time.extend([compute_time, 0.0])
        elif experts_parallelism == "tp":
            # we care only about per GPU stats
            parallel_size = n_local_gpus if experts_are_intra else total_n_gpus
            compute_time = max(
                n_act_experts * expert_param_bytes / parallel_size / gpu_mem_bw,
                batch_size
                * prompt_len
                * top_k
                * expert_flops
                / parallel_size
                / gpu_flops,
            )
            comm_time = (
                intra_node_comm_time if experts_are_intra else inter_node_comm_time
            )
            exec_time.extend([compute_time, comm_time])
        elif experts_parallelism == "ep":
            n_experts_per_local_gpu = (
                n_experts if experts_are_intra else experts_allocation[node_idx]
            ) // n_local_gpus
            compute_time = max(
                ceildiv(n_act_experts * n_experts_per_local_gpu, n_experts)
                * expert_param_bytes
                / gpu_mem_bw,
                ceildiv(
                    batch_size * prompt_len * top_k * n_experts_per_local_gpu, n_experts
                )
                * expert_flops
                / gpu_flops,
            )
            comm_time = (
                intra_node_comm_time if experts_are_intra else inter_node_comm_time
            )
            exec_time.extend([compute_time, comm_time])
        elif experts_parallelism == "ep+tp":
            assert not experts_are_intra
            n_local_experts = experts_allocation[node_idx]
            compute_time = max(
                ceildiv(n_act_experts * n_local_experts, n_experts)
                * expert_param_bytes
                / n_local_gpus
                / gpu_mem_bw,
                ceildiv(batch_size * prompt_len * top_k * n_local_experts, n_experts)
                * expert_flops
                / n_local_gpus
                / gpu_flops,
            )
            exec_time.extend([compute_time, inter_node_comm_time])

        constant = pp_node_layers[node_idx] if pp_strategy else n_layers
        exec_time = [val * constant for val in exec_time]

        extra_comm_time = 0.0
        if pp_is_naive:
            extra_comm_time += (intra_p2p_oh + comm_data_size / intra_p2p_bw) * (
                n_local_gpus - 1
            )
        if pp_strategy:
            if node_idx < len(SETUP) - 1:
                extra_comm_time += inter_p2p_oh + comm_data_size / inter_p2p_bw
            else:
                out_size = precision_bytes * batch_size * prompt_len * vocab_d
                extra_comm_time += inter_coll_oh + out_size / inter_coll_bw

        exec_time.append(extra_comm_time)
        exec_time_by_node.append(exec_time)

    return exec_time_by_node
