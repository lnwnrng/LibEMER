import os

import numpy as np
import torch
import random

from config.setting import resolve_effective_experiment_mode, resolve_effective_split_type
from utils.experiment_logger import (
    log_split_info,
    make_log_context,
    metric_order,
    print_section,
    print_table,
)
from utils.store import save_res


def state_log(args):
    print_section("Experiment", [
        ("model", args.model),
        ("dataset", args.dataset),
        ("setting", args.setting),
        ("seed", args.seed),
        ("device", args.device),
    ])
    print_section("Data And Preprocess", [
        ("dataset path", args.dataset_path),
        ("feature type", args.feature_type),
        ("pass band", [args.low_pass, args.high_pass]),
        ("time window", args.time_window),
        ("overlap", args.overlap),
        ("sample length", args.sample_length),
        ("stride", args.stride),
        ("normalize", args.normalize),
        ("label used", args.label_used),
    ])
    print_section("Split", [
        ("experiment mode", resolve_effective_experiment_mode(args)),
        ("split type", resolve_effective_split_type(args)),
        ("fold num", args.fold_num),
        ("front", args.front),
        ("sessions", args.sessions),
        ("test size", args.test_size),
        ("val size", args.val_size),
    ])
    print_section("Training", [
        ("batch size", args.batch_size),
        ("epochs", args.epochs),
        ("learning rate", args.lr),
        ("loss func", args.loss_func),
        ("metrics", args.metrics),
        ("metric choose", args.metric_choose),
    ])
    print_section("Output", [
        ("log dir", args.log_dir),
        ("output dir", args.output_dir),
    ])


def result_log(args, best_metrics):
    output = {}
    headers = ["round"]
    ordered_metrics = metric_order(args.metrics)
    std_flags = {}
    for metric_name in ordered_metrics:
        flag = any((metric_name+"_std") in metric for metric in best_metrics)
        std_flags[metric_name] = flag
        output[metric_name] = []
        headers.append(metric_name + ("_mean" if flag else ""))
        if flag:
            headers.append(metric_name + "_std")
    rows = []
    for idx, metric in enumerate(best_metrics):
        row = [idx + 1]
        for n in ordered_metrics:
            output[n].append(metric[n])
            row.append(metric[n])
            if std_flags[n]:
                row.append(metric.get(n+"_std"))
        rows.append(row)
    print_table("Round Results", headers, rows)

    summary_rows = []
    for metric in ordered_metrics:
        summary_rows.append(["all rounds", metric, np.mean(output[metric]), np.std(output[metric])])
        save_res(args, "ALLRound Mean and Std of {} : {:.4f}/{:.4f}".format(metric, np.mean(output[metric]), np.std(output[metric])))
    print_table("Final Result Summary", ["scope", "metric", "mean", "std"], summary_rows)

def sub_result_log(args, subjects_metrics):
    sub_outputs = {}
    for i, sub_metric in enumerate(subjects_metrics):
        sub_output = {}
        for metric in args.metrics:
            sub_output[metric] = 0
            for r_metric in sub_metric:
                sub_output[metric] += r_metric[metric]
            sub_output[metric] /= len(subjects_metrics[i])
        sub_outputs[f"sub {i}"] = sub_output
    # sub_outputs: (sub, metric)
    save_res(args, sub_outputs)
    sub_mean_std = {}
    summary_rows = []
    for metric in metric_order(args.metrics):
        sub_metrics = []
        for sub_metric in sub_outputs.values():
            sub_metrics.append(sub_metric[metric])
        sub_mean_std[metric] = {"mean": np.mean(sub_metrics), "std": np.std(sub_metrics)}
        summary_rows.append(["all subjects", metric, np.mean(sub_metrics), np.std(sub_metrics)])
        save_res(args,f"ALL Subjects {metric}: Mean: {np.mean(sub_metrics)}, Std: {np.std(sub_metrics)}")
    print_table("Final Result Summary", ["scope", "metric", "mean", "std"], summary_rows)


def split_log(train_indexes=None, test_indexes=None, val_indexes=None, train_data=None, val_data=None,
              test_data=None, test_sub_label=None, context=None):
    log_split_info(
        train_indexes=train_indexes,
        test_indexes=test_indexes,
        val_indexes=val_indexes,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        test_sub_label=test_sub_label,
        context=context,
    )




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False
