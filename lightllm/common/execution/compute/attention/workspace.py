def lane_workspace(workspace, limits, index):
    if not 0 <= index < limits.microbatch_count:
        raise ValueError("microbatch exceeds reserved execution lanes")
    if workspace is None:
        return None
    size = workspace.numel() // limits.microbatch_count
    return workspace[index * size : (index + 1) * size]
