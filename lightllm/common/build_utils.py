
def repair_config(config, same_names):
    find_value = None
    for name in same_names:
        if name in config and config[name] is not None:
            find_value = config[name]
            break
    for name in same_names:
        config[name] = find_value
    return