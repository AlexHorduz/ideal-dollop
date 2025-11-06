import json
import optuna
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from tuning.train import train


def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def plot_training_validation_results(train_history, val_history, save_path):
    if not train_history:
        return

    has_val_history = bool(val_history)
    n_cols = 2 if has_val_history else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(12, 5))
    if n_cols == 1:
        axes = np.atleast_1d(axes)

    ax_loss = axes[0]
    train_epochs = [entry['epoch'] for entry in train_history]
    train_losses = [entry['total_loss'] for entry in train_history]
    ax_loss.plot(train_epochs, train_losses, marker='o', label='Train loss')

    if has_val_history:
        val_epochs = [entry['epoch'] for entry in val_history]
        val_losses = [entry['loss'] for entry in val_history]
        ax_loss.plot(val_epochs, val_losses, marker='s', label='Validation loss')

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss Dynamics')
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    if has_val_history:
        ax_metrics = axes[1]
        val_epochs = [entry['epoch'] for entry in val_history]
        metric_keys = [('mAP', 'mAP'), ('precision', 'Precision'),
                       ('recall', 'Recall'), ('f1_score', 'F1-score')]

        for key, label in metric_keys:
            epochs_metric = [entry['epoch'] for entry in val_history if entry.get(key) is not None]
            values = [entry.get(key) for entry in val_history if entry.get(key) is not None]
            if values:
                ax_metrics.plot(epochs_metric, values, marker='o', label=label)

        ax_metrics.set_xlabel('Epoch')
        ax_metrics.set_ylabel('Metric value')
        ax_metrics.set_title('Validation Metrics')
        ax_metrics.set_ylim(0, 1.05)
        ax_metrics.grid(alpha=0.3)
        ax_metrics.legend()

    fig.suptitle('Training Progress Overview')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_test_summary(test_summary, save_path):
    if not test_summary:
        return

    metrics = ['mAP', 'precision', 'recall', 'f1_score']
    values = [test_summary.get(metric) for metric in metrics if test_summary.get(metric) is not None]
    labels = [metric.upper() if metric == 'mAP' else metric.capitalize() for metric in metrics if test_summary.get(metric) is not None]

    if not values:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Test Set Performance')
    ax.grid(alpha=0.2, axis='y')
    for idx, val in enumerate(values):
        ax.text(idx, val + 0.02, f"{val:.3f}", ha='center', va='bottom')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_val_test_comparison(val_summary, test_summary, save_path):
    if not val_summary or not test_summary:
        return

    metrics = ['mAP', 'precision', 'recall', 'f1_score']
    val_values = [val_summary.get(metric) for metric in metrics]
    test_values = [test_summary.get(metric) for metric in metrics]

    if any(v is None for v in val_values) or any(v is None for v in test_values):
        return

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, val_values, width, label='Validation')
    ax.bar(x + width / 2, test_values, width, label='Test')

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() if m == 'mAP' else m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Validation vs Test Metrics')
    ax.legend()
    ax.grid(alpha=0.2, axis='y')

    for idx, (v_val, v_test) in enumerate(zip(val_values, test_values)):
        ax.text(idx - width / 2, v_val + 0.02, f"{v_val:.3f}", ha='center', va='bottom')
        ax.text(idx + width / 2, v_test + 0.02, f"{v_test:.3f}", ha='center', va='bottom')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_trial_metric_overview(trials, metric_key, save_path, ylabel):
    points = [(t['number'], t['metrics'].get(metric_key)) for t in trials if t['metrics'].get(metric_key) is not None]
    if not points:
        return

    trial_numbers, values = zip(*points)

    plt.figure(figsize=(10, 5))
    plt.plot(trial_numbers, values, 'o-', alpha=0.7)
    plt.xlabel('Trial')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} per Trial')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def save_trial_artifacts(model_name, trial_number, results):
    trial_dir = Path("results") / model_name / "trials" / f"trial_{trial_number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    train_history = results.get('train_history')
    val_history = results.get('val_history')
    test_summary = results.get('test_summary')
    val_summary = results.get('validation_summary')

    if train_history is not None:
        train_history_path = trial_dir / 'train_history.json'
        save_json(train_history, train_history_path)
        artifacts['train_history_path'] = str(train_history_path)

    if val_history is not None:
        val_history_path = trial_dir / 'val_history.json'
        save_json(val_history, val_history_path)
        artifacts['val_history_path'] = str(val_history_path)

    metrics_plot_path = trial_dir / 'training_validation.png'
    plot_training_validation_results(train_history or [], val_history or [], metrics_plot_path)
    if metrics_plot_path.exists():
        artifacts['training_validation_plot'] = str(metrics_plot_path)

    if test_summary:
        test_summary_path = trial_dir / 'test_summary.json'
        save_json(test_summary, test_summary_path)
        artifacts['test_summary_path'] = str(test_summary_path)

        test_plot_path = trial_dir / 'test_metrics.png'
        plot_test_summary(test_summary, test_plot_path)
        if test_plot_path.exists():
            artifacts['test_plot_path'] = str(test_plot_path)

        comparison_plot_path = trial_dir / 'validation_vs_test.png'
        plot_val_test_comparison(val_summary, test_summary, comparison_plot_path)
        if comparison_plot_path.exists():
            artifacts['val_test_plot_path'] = str(comparison_plot_path)

    return artifacts


def create_objective(model_name):
    def objective(trial):
        config_override = {
            'lr': trial.suggest_categorical('lr', cfg.HYPERPARAMETER_SEARCH_SPACE['lr']),
            'batch_size': trial.suggest_categorical('batch_size', cfg.HYPERPARAMETER_SEARCH_SPACE['batch_size']),
            'epochs': trial.suggest_categorical('epochs', cfg.HYPERPARAMETER_SEARCH_SPACE['epochs']),
            'lambda_box': trial.suggest_categorical('lambda_box', cfg.HYPERPARAMETER_SEARCH_SPACE['lambda_box']),
            'lambda_obj': trial.suggest_categorical('lambda_obj', cfg.HYPERPARAMETER_SEARCH_SPACE['lambda_obj']),
            'lambda_cls': trial.suggest_categorical('lambda_cls', cfg.HYPERPARAMETER_SEARCH_SPACE['lambda_cls']),
            'optimizer': trial.suggest_categorical('optimizer', cfg.HYPERPARAMETER_SEARCH_SPACE['optimizer']),
            'weight_decay': trial.suggest_categorical('weight_decay', cfg.HYPERPARAMETER_SEARCH_SPACE['weight_decay']),
            'augment': trial.suggest_categorical('augment', cfg.HYPERPARAMETER_SEARCH_SPACE['augment']),
        }

        print(f"Trial {trial.number}: {config_override}")
        results = train(model_name, config_override, trial=trial)

        # Store additional metrics in trial user attributes
        trial.set_user_attr('mAP', float(results['mAP']))
        trial.set_user_attr('precision', float(results['precision'].mean()))
        trial.set_user_attr('recall', float(results['recall'].mean()))
        trial.set_user_attr('f1_score', float(results['f1_score']))
        trial.set_user_attr('loss', float(results['loss']))
        trial.set_user_attr('box_loss', float(results['box_loss']))
        trial.set_user_attr('obj_loss', float(results['obj_loss']))
        trial.set_user_attr('noobj_loss', float(results['noobj_loss']))
        trial.set_user_attr('cls_loss', float(results['cls_loss']))

        train_history = results.get('train_history', [])
        val_history = results.get('val_history', [])
        test_summary = results.get('test_summary')

        if train_history:
            trial.set_user_attr('train_final_loss', float(train_history[-1]['total_loss']))
        if val_history:
            last_val = val_history[-1]
            trial.set_user_attr('val_loss', float(last_val['loss']))
            trial.set_user_attr('val_mAP', float(last_val.get('mAP', results['mAP'])))
            trial.set_user_attr('val_f1', float(last_val.get('f1_score', results['f1_score'])))
        if test_summary:
            trial.set_user_attr('test_mAP', float(test_summary['mAP']))
            trial.set_user_attr('test_f1', float(test_summary['f1_score']))
            trial.set_user_attr('test_precision', float(test_summary['precision']))
            trial.set_user_attr('test_recall', float(test_summary['recall']))
            trial.set_user_attr('test_loss', float(test_summary['loss']))

        artifacts = save_trial_artifacts(model_name, trial.number, results)
        for key, path in artifacts.items():
            trial.set_user_attr(key, path)

        # Optimize for combined metric: 0.7 * mAP + 0.3 * F1
        # This balances detection accuracy (mAP) with precision-recall trade-off (F1)
        combined_metric = 0.7 * results['mAP'] + 0.3 * results['f1_score']
        trial.set_user_attr('combined_metric', float(combined_metric))

        return combined_metric

    return objective


def plot_optimization_history(study, save_path):
    plt.figure(figsize=(10, 6))

    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    trial_numbers = [t.number for t in trials if t.value is not None]

    plt.plot(trial_numbers, values, 'o-', alpha=0.6)
    plt.xlabel('Trial')
    plt.ylabel('Combined Metric (0.7*mAP + 0.3*F1)')
    plt.title('Optimization History')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_param_importance(study, save_path):
    importance = optuna.importance.get_param_importances(study)

    params = list(importance.keys())
    values = list(importance.values())

    sorted_indices = np.argsort(values)[::-1]
    params = [params[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(params, values)
    plt.xlabel('Importance')
    plt.ylabel('Parameter')
    plt.title('Hyperparameter Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

    return importance


def tune_model(model_name, n_trials=20):
    print(f"Starting hyperparameter tuning for {model_name.upper()}")
    print(f"Number of trials: {n_trials}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    model_dir = results_dir / model_name
    model_dir.mkdir(exist_ok=True)

    study = optuna.create_study(
        direction='maximize',
        study_name=f'{model_name}_tuning',
        sampler=optuna.samplers.TPESampler(multivariate=True, constant_liar=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10)
    )

    objective = create_objective(model_name)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial

    print(f"\nTuning completed for {model_name.upper()}")
    print(f"Best Combined Metric: {best_value:.4f} (0.7*mAP + 0.3*F1)")
    print(f"Best mAP: {best_trial.user_attrs.get('mAP', 0):.4f}")
    print(f"Best F1-Score: {best_trial.user_attrs.get('f1_score', 0):.4f}")
    print(f"Best Precision: {best_trial.user_attrs.get('precision', 0):.4f}")
    print(f"Best Recall: {best_trial.user_attrs.get('recall', 0):.4f}")
    print(f"Best parameters: {best_params}")

    best_test_map = best_trial.user_attrs.get('test_mAP')
    best_test_f1 = best_trial.user_attrs.get('test_f1')
    best_test_precision = best_trial.user_attrs.get('test_precision')
    best_test_recall = best_trial.user_attrs.get('test_recall')
    best_test_loss = best_trial.user_attrs.get('test_loss')

    if best_test_map is not None:
        print(f"Best Test mAP: {best_test_map:.4f}")
    if best_test_f1 is not None:
        print(f"Best Test F1-Score: {best_test_f1:.4f}")
    if best_test_precision is not None and best_test_recall is not None:
        print(f"Best Test Precision/Recall: {best_test_precision:.4f} / {best_test_recall:.4f}")
    if best_test_loss is not None:
        print(f"Best Test Loss: {best_test_loss:.4f}")

    all_trials = []
    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': str(trial.state),
            'metrics': {
                'mAP': trial.user_attrs.get('mAP'),
                'f1_score': trial.user_attrs.get('f1_score'),
                'precision': trial.user_attrs.get('precision'),
                'recall': trial.user_attrs.get('recall'),
                'combined_metric': trial.user_attrs.get('combined_metric'),
                'loss': trial.user_attrs.get('loss'),
                'box_loss': trial.user_attrs.get('box_loss'),
                'obj_loss': trial.user_attrs.get('obj_loss'),
                'noobj_loss': trial.user_attrs.get('noobj_loss'),
                'cls_loss': trial.user_attrs.get('cls_loss'),
                'train_final_loss': trial.user_attrs.get('train_final_loss'),
                'val_loss': trial.user_attrs.get('val_loss'),
                'val_mAP': trial.user_attrs.get('val_mAP'),
                'val_f1': trial.user_attrs.get('val_f1'),
                'test_mAP': trial.user_attrs.get('test_mAP'),
                'test_f1': trial.user_attrs.get('test_f1'),
                'test_precision': trial.user_attrs.get('test_precision'),
                'test_recall': trial.user_attrs.get('test_recall'),
                'test_loss': trial.user_attrs.get('test_loss')
            }
        }
        artifact_keys = [
            'train_history_path',
            'val_history_path',
            'test_summary_path',
            'training_validation_plot',
            'test_plot_path',
            'val_test_plot_path'
        ]
        trial_data['artifacts'] = {
            key: trial.user_attrs.get(key)
            for key in artifact_keys
            if trial.user_attrs.get(key) is not None
        }
        all_trials.append(trial_data)

    plot_trial_metric_overview(all_trials, 'mAP', model_dir / 'trials_validation_mAP.png', 'Validation mAP')
    plot_trial_metric_overview(all_trials, 'f1_score', model_dir / 'trials_validation_F1.png', 'Validation F1-score')
    plot_trial_metric_overview(all_trials, 'test_mAP', model_dir / 'trials_test_mAP.png', 'Test mAP')
    plot_trial_metric_overview(all_trials, 'test_f1', model_dir / 'trials_test_F1.png', 'Test F1-score')

    importance = optuna.importance.get_param_importances(study)
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    results = {
        'model': model_name,
        'n_trials': n_trials,
        'best_combined_metric': best_value,
        'best_mAP': best_trial.user_attrs.get('mAP'),
        'best_f1_score': best_trial.user_attrs.get('f1_score'),
        'best_precision': best_trial.user_attrs.get('precision'),
        'best_recall': best_trial.user_attrs.get('recall'),
        'best_loss': best_trial.user_attrs.get('loss'),
        'best_box_loss': best_trial.user_attrs.get('box_loss'),
        'best_obj_loss': best_trial.user_attrs.get('obj_loss'),
        'best_noobj_loss': best_trial.user_attrs.get('noobj_loss'),
        'best_cls_loss': best_trial.user_attrs.get('cls_loss'),
        'best_val_loss': best_trial.user_attrs.get('val_loss'),
        'best_train_final_loss': best_trial.user_attrs.get('train_final_loss'),
        'best_test_mAP': best_trial.user_attrs.get('test_mAP'),
        'best_test_f1_score': best_trial.user_attrs.get('test_f1'),
        'best_test_precision': best_trial.user_attrs.get('test_precision'),
        'best_test_recall': best_trial.user_attrs.get('test_recall'),
        'best_test_loss': best_trial.user_attrs.get('test_loss'),
        'best_params': best_params,
        'best_train_history_path': best_trial.user_attrs.get('train_history_path'),
        'best_val_history_path': best_trial.user_attrs.get('val_history_path'),
        'best_test_summary_path': best_trial.user_attrs.get('test_summary_path'),
        'best_training_plot_path': best_trial.user_attrs.get('training_validation_plot'),
        'best_test_plot_path': best_trial.user_attrs.get('test_plot_path'),
        'best_val_test_plot_path': best_trial.user_attrs.get('val_test_plot_path'),
        'all_trials': all_trials,
        'param_importance': sorted_importance,
        'param_importance_ranked': list(sorted_importance.keys())
    }

    save_json(results, results_dir / f'{model_name}_tuning.json')
    save_json(best_params, results_dir / f'{model_name}_best.json')
    save_json(sorted_importance, results_dir / f'{model_name}_importance.json')

    print(f"Creating visualizations...")
    plot_optimization_history(study, results_dir / f'{model_name}_optimization_history.png')
    plot_param_importance(study, results_dir / f'{model_name}_param_importance.png')

    fig1 = optuna.visualization.plot_param_importances(study)
    fig1.write_image(results_dir / f'{model_name}_optuna_importance.png')
    print(f"Saved: {results_dir / f'{model_name}_optuna_importance.png'}")

    fig2 = optuna.visualization.plot_optimization_history(study)
    fig2.write_image(results_dir / f'{model_name}_optuna_history.png')
    print(f"Saved: {results_dir / f'{model_name}_optuna_history.png'}")

    fig3 = optuna.visualization.plot_slice(study)
    fig3.write_image(results_dir / f'{model_name}_slice_plot.png')
    print(f"Saved: {results_dir / f'{model_name}_slice_plot.png'}")

    print(f"All results saved to: {results_dir}/")

    return results


def create_comparison_report(yolo_results, alexnet_results):
    results_dir = Path("results")

    comparison = {
        'yolo': {
            'best_combined_metric': yolo_results['best_combined_metric'],
            'best_mAP': yolo_results['best_mAP'],
            'best_f1_score': yolo_results['best_f1_score'],
            'best_precision': yolo_results['best_precision'],
            'best_recall': yolo_results['best_recall'],
            'best_loss': yolo_results['best_loss'],
            'best_box_loss': yolo_results['best_box_loss'],
            'best_obj_loss': yolo_results['best_obj_loss'],
            'best_noobj_loss': yolo_results['best_noobj_loss'],
            'best_cls_loss': yolo_results['best_cls_loss'],
            'best_val_loss': yolo_results.get('best_val_loss'),
            'best_train_final_loss': yolo_results.get('best_train_final_loss'),
            'best_test_mAP': yolo_results.get('best_test_mAP'),
            'best_test_f1_score': yolo_results.get('best_test_f1_score'),
            'best_test_precision': yolo_results.get('best_test_precision'),
            'best_test_recall': yolo_results.get('best_test_recall'),
            'best_test_loss': yolo_results.get('best_test_loss'),
            'best_params': yolo_results['best_params'],
            'top_5_important_params': yolo_results['param_importance_ranked'][:5],
            'best_training_plot_path': yolo_results.get('best_training_plot_path'),
            'best_test_plot_path': yolo_results.get('best_test_plot_path'),
            'best_val_test_plot_path': yolo_results.get('best_val_test_plot_path')
        },
        'alexnet': {
            'best_combined_metric': alexnet_results['best_combined_metric'],
            'best_mAP': alexnet_results['best_mAP'],
            'best_f1_score': alexnet_results['best_f1_score'],
            'best_precision': alexnet_results['best_precision'],
            'best_recall': alexnet_results['best_recall'],
            'best_loss': alexnet_results['best_loss'],
            'best_box_loss': alexnet_results['best_box_loss'],
            'best_obj_loss': alexnet_results['best_obj_loss'],
            'best_noobj_loss': alexnet_results['best_noobj_loss'],
            'best_cls_loss': alexnet_results['best_cls_loss'],
            'best_val_loss': alexnet_results.get('best_val_loss'),
            'best_train_final_loss': alexnet_results.get('best_train_final_loss'),
            'best_test_mAP': alexnet_results.get('best_test_mAP'),
            'best_test_f1_score': alexnet_results.get('best_test_f1_score'),
            'best_test_precision': alexnet_results.get('best_test_precision'),
            'best_test_recall': alexnet_results.get('best_test_recall'),
            'best_test_loss': alexnet_results.get('best_test_loss'),
            'best_params': alexnet_results['best_params'],
            'top_5_important_params': alexnet_results['param_importance_ranked'][:5],
            'best_training_plot_path': alexnet_results.get('best_training_plot_path'),
            'best_test_plot_path': alexnet_results.get('best_test_plot_path'),
            'best_val_test_plot_path': alexnet_results.get('best_val_test_plot_path')
        }
    }

    save_json(comparison, results_dir / 'comparison.json')

    models = ['YOLO', 'AlexNet']
    val_maps = [yolo_results['best_mAP'], alexnet_results['best_mAP']]
    test_maps = [yolo_results.get('best_test_mAP'), alexnet_results.get('best_test_mAP')]
    val_f1 = [yolo_results['best_f1_score'], alexnet_results['best_f1_score']]
    test_f1 = [yolo_results.get('best_test_f1_score'), alexnet_results.get('best_test_f1_score')]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, val_maps, width, label='Validation mAP', color='#1f77b4')
    test_map_values = [tm if tm is not None else 0.0 for tm in test_maps]
    if any(tm is not None for tm in test_maps):
        ax.bar(x + width / 2, test_map_values, width, label='Test mAP', color='#ff7f0e')

    max_map = max(val_maps + [tm for tm in test_maps if tm is not None] or val_maps)
    ax.set_ylabel('mAP@[0.5:0.95]')
    ax.set_title('Validation vs Test mAP')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, max_map * 1.2 if max_map > 0 else 1)
    ax.grid(alpha=0.2, axis='y')
    ax.legend()

    for idx, val in enumerate(val_maps):
        ax.text(x[idx] - width / 2, val + 0.01, f'{val:.4f}', ha='center', va='bottom')
    for idx, val in enumerate(test_maps):
        if val is not None:
            ax.text(x[idx] + width / 2, val + 0.01, f'{val:.4f}', ha='center', va='bottom')
        else:
            ax.text(x[idx] + width / 2, 0.02, 'N/A', ha='center', va='bottom')

    fig.tight_layout()
    map_plot_path = results_dir / 'model_comparison_map.png'
    fig.savefig(map_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {map_plot_path}")

    fig_f1, ax_f1 = plt.subplots(figsize=(10, 6))
    ax_f1.bar(x - width / 2, val_f1, width, label='Validation F1', color='#2ca02c')
    test_f1_values = [tf if tf is not None else 0.0 for tf in test_f1]
    if any(tf is not None for tf in test_f1):
        ax_f1.bar(x + width / 2, test_f1_values, width, label='Test F1', color='#d62728')

    max_f1 = max(val_f1 + [tf for tf in test_f1 if tf is not None] or val_f1)
    ax_f1.set_ylabel('F1-score')
    ax_f1.set_title('Validation vs Test F1-score')
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(models)
    ax_f1.set_ylim(0, max_f1 * 1.2 if max_f1 > 0 else 1)
    ax_f1.grid(alpha=0.2, axis='y')
    ax_f1.legend()

    for idx, val in enumerate(val_f1):
        ax_f1.text(x[idx] - width / 2, val + 0.01, f'{val:.4f}', ha='center', va='bottom')
    for idx, val in enumerate(test_f1):
        if val is not None:
            ax_f1.text(x[idx] + width / 2, val + 0.01, f'{val:.4f}', ha='center', va='bottom')
        else:
            ax_f1.text(x[idx] + width / 2, 0.02, 'N/A', ha='center', va='bottom')

    fig_f1.tight_layout()
    f1_plot_path = results_dir / 'model_comparison_f1.png'
    fig_f1.savefig(f1_plot_path, dpi=150)
    plt.close(fig_f1)
    print(f"Saved: {f1_plot_path}")

    print(f"YOLO Results:")
    print(f"Combined Metric (0.7*mAP + 0.3*F1): {yolo_results['best_combined_metric']:.4f}")
    print(f"Best mAP@[0.5:0.95]: {yolo_results['best_mAP']:.4f}")
    print(f"Best F1-Score: {yolo_results['best_f1_score']:.4f}")
    print(f"Best Precision: {yolo_results['best_precision']:.4f}")
    print(f"Best Recall: {yolo_results['best_recall']:.4f}")
    print(f"Best Loss: {yolo_results['best_loss']:.4f} (Box: {yolo_results['best_box_loss']:.4f}, Obj: {yolo_results['best_obj_loss']:.4f}, NoObj: {yolo_results['best_noobj_loss']:.4f}, Cls: {yolo_results['best_cls_loss']:.4f})")
    print(f"Important params: {yolo_results['param_importance_ranked']}")
    print(f"Best params: {yolo_results['best_params']}")
    if yolo_results.get('best_test_mAP') is not None:
        print(f"Test mAP@[0.5:0.95]: {yolo_results['best_test_mAP']:.4f}")
        print(f"Test F1-Score: {yolo_results['best_test_f1_score']:.4f}")
        print(f"Test Precision: {yolo_results['best_test_precision']:.4f}")
        print(f"Test Recall: {yolo_results['best_test_recall']:.4f}")
        print(f"Test Loss: {yolo_results['best_test_loss']:.4f}")

    print(f"AlexNet Results:")
    print(f"Combined Metric (0.7*mAP + 0.3*F1): {alexnet_results['best_combined_metric']:.4f}")
    print(f"Best mAP@[0.5:0.95]: {alexnet_results['best_mAP']:.4f}")
    print(f"Best F1-Score: {alexnet_results['best_f1_score']:.4f}")
    print(f"Best Precision: {alexnet_results['best_precision']:.4f}")
    print(f"Best Recall: {alexnet_results['best_recall']:.4f}")
    print(f"Best Loss: {alexnet_results['best_loss']:.4f} (Box: {alexnet_results['best_box_loss']:.4f}, Obj: {alexnet_results['best_obj_loss']:.4f}, NoObj: {alexnet_results['best_noobj_loss']:.4f}, Cls: {alexnet_results['best_cls_loss']:.4f})")
    print(f"Important params: {alexnet_results['param_importance_ranked']}")
    print(f"Best params: {alexnet_results['best_params']}")
    if alexnet_results.get('best_test_mAP') is not None:
        print(f"Test mAP@[0.5:0.95]: {alexnet_results['best_test_mAP']:.4f}")
        print(f"Test F1-Score: {alexnet_results['best_test_f1_score']:.4f}")
        print(f"Test Precision: {alexnet_results['best_test_precision']:.4f}")
        print(f"Test Recall: {alexnet_results['best_test_recall']:.4f}")
        print(f"Test Loss: {alexnet_results['best_test_loss']:.4f}")
    
    save_json(alexnet_results, results_dir / 'alexnet_results.json')
    save_json(yolo_results, results_dir / 'yolo_results.json')


if __name__ == "__main__":
    n_trials = 4
    yolo_results = tune_model('yolo', n_trials=n_trials)
    alexnet_results = tune_model('alexnet', n_trials=n_trials)
    create_comparison_report(yolo_results, alexnet_results)
