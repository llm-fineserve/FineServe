def get_rank(node_rank, nproc_per_node, local_rank):
    return node_rank * nproc_per_node + local_rank


def get_manager_port(base_port: int, local_rank: int):
    return base_port + local_rank * 2   # one for REP/REQ, one for PUB/SUB
