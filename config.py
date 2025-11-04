ALEXNET_CONFIG = {
    # Model architecture
    'model_type': 'alexnet',
    'num_classes': 15,
    'grid_size': 7,
    'B': 2,  # number of bounding boxes per grid cell

    # Optimization
    'lr': 1e-3,
    'batch_size': 32,
    'epochs': 5,
    'optimizer': 'adam',
    'weight_decay': 5e-4,

    # Loss weights
    'lambda_box': 0.05,  # box localization loss weight
    'lambda_obj': 1.0,  # objectness loss weight
    'lambda_noobj': 0.5,  # no-objectness loss weight
    'lambda_cls': 0.5,  # classification loss weight

    # Data
    'input_size': (224, 224),
    'augment': True,

    # Data augmentation params
    'hflip_prob': 0.5,
    'color_jitter_brightness': 0.2,
    'color_jitter_contrast': 0.2,
    'color_jitter_saturation': 0.2,
    'color_jitter_hue': 0.02,
}

YOLO_CONFIG = {
    'model_type': 'yolo',
    'num_classes': 15,
    'grid_size': 7,
    'B': 2,  # number of bounding boxes per grid cell

    # Optimization
    'lr': 1e-4,
    'batch_size': 32,
    'epochs': 5,
    'optimizer': 'adamw',
    'weight_decay': 1e-4,

    # Loss weights
    'lambda_box': 0.05,  # box localization loss weight
    'lambda_obj': 1.0,  # objectness loss weight
    'lambda_noobj': 0.5,  # no-objectness loss weight
    'lambda_cls': 0.5,  # classification loss weight

    # Data
    'input_size': (448, 448),
    'augment': True,

    # Data augmentation params
    'hflip_prob': 0.5,
    'color_jitter_brightness': 0.2,
    'color_jitter_contrast': 0.2,
    'color_jitter_saturation': 0.2,
    'color_jitter_hue': 0.02,
}

# Evaluation configuration
EVAL_CONFIG = {
    'iou_thresholds': [0.5], 
    'iou_thr_pr': 0.5,  # IoU threshold for Precision/Recall
    'n_conf_thresholds': 50,  # number of confidence thresholds for PR curve
    'conf_threshold': 0.1,  # confidence threshold for inference (lowered from 0.25 for early training)
    'nms_iou_threshold': 0.45,  # IoU threshold for NMS
}

# Training configuration
TRAINING_CONFIG = {
    'num_workers': 8,
    'pin_memory': False,  # Edit True for CUDA
    'save_every': 10,  # save checkpoint every N epochs
    'eval_every': 10,  # evaluate every N epochs
    'grad_clip_norm': 10.0,  # gradient clipping
}

HYPERPARAMETER_SEARCH_SPACE = {
    'lr': [1e-4, 5e-4, 1e-3],  # learning rate
    'epochs': [20],  # number of epochs,
    'batch_size': [32],  # batch size
    'lambda_box': [0.05],  # box loss weight
    'lambda_obj': [1.0],  # objectness loss weight
    'lambda_cls': [0.5],  # classification loss weight
    'optimizer': ['adam'],  # optimizer type
    'weight_decay': [1e-5, 1e-4, 5e-4],  # L2 regularization,
    'augment': [True],  # data augmentation
}


def get_config(model_name: str) -> dict:
    if model_name.lower() == 'alexnet':
        return ALEXNET_CONFIG.copy()
    elif model_name.lower() == 'yolo':
        return YOLO_CONFIG.copy()
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'alexnet' or 'yolo'.")


def generate_hyperparameter_configs(model_name: str, param_name: str, base_config: dict = None) -> list:
    if base_config is None:
        base_config = get_config(model_name)

    if param_name not in HYPERPARAMETER_SEARCH_SPACE:
        raise ValueError(f"Parameter '{param_name}' not in search space. "
                        f"Available: {list(HYPERPARAMETER_SEARCH_SPACE.keys())}")

    configs = []
    for value in HYPERPARAMETER_SEARCH_SPACE[param_name]:
        config = base_config.copy()
        config[param_name] = value
        config['experiment_name'] = f"{model_name}_{param_name}={value}"
        configs.append(config)

    return configs


def get_all_hyperparameter_configs(model_name: str) -> dict:
    experiments = {}
    base_config = get_config(model_name)

    tunable_params = [p for p in HYPERPARAMETER_SEARCH_SPACE.keys()]

    for param in tunable_params:
        experiments[param] = generate_hyperparameter_configs(model_name, param, base_config)

    return experiments


if __name__ == "__main__":
    import itertools

    param_names = list(HYPERPARAMETER_SEARCH_SPACE.keys())
    param_values = [HYPERPARAMETER_SEARCH_SPACE[param] for param in param_names]
    all_combinations = list(itertools.product(*param_values))

    print(f"\nTotal possible combinations: {len(all_combinations)}")
    print(f"Parameters: {', '.join(param_names)}")

    for i, combination in enumerate(all_combinations, 1):
        params = {param_names[j]: combination[j] for j in range(len(param_names))}
        print(f"Random Combination {i}:")
        for param, value in params.items():
            print(f"  {param:20s}: {value}")
        print()
