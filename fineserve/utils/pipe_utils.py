
def pipeline_split(num_hidden_layers: int,
                   pipeline_parallel_size: int):
    num_layer_per_part = num_hidden_layers // pipeline_parallel_size
    partition = [num_layer_per_part] * pipeline_parallel_size
    if num_hidden_layers % pipeline_parallel_size == pipeline_parallel_size - 1:
        start_partition = 0
    else:
        start_partition = 1
    for i in range(num_hidden_layers % pipeline_parallel_size):
        partition[start_partition + i] += 1
    return partition
