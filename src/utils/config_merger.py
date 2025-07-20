def config_merge(config1, config2):
    for key, value in config2.items():
        if isinstance(value, dict):
            config1[key] = config_merge(config1.get(key, {}), value)
        else:
            config1[key] = value
    return config1