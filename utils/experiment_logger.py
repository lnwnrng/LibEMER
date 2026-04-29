import csv
import os
from pathlib import Path


PRIMARY_METRICS = ("acc", "macro-f1", "micro-f1", "weighted-f1", "f1", "ck", "loss")


def format_value(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_lr(value):
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def metric_order(metric_names):
    ordered = [name for name in PRIMARY_METRICS if name in metric_names]
    ordered.extend(name for name in metric_names if name not in ordered)
    return ordered


def get_current_lr(*optimizers):
    lrs = []
    for optimizer in optimizers:
        if optimizer is None:
            continue
        for group in getattr(optimizer, "param_groups", []):
            if "lr" in group:
                lrs.append(group["lr"])
    if not lrs:
        return None

    unique = []
    for lr in lrs:
        if lr not in unique:
            unique.append(lr)
    if len(unique) == 1:
        return unique[0]
    return "/".join(format_lr(lr) for lr in unique)


def print_section(title, rows=None, width=78):
    print("=" * width)
    print(title.upper())
    print("-" * width)
    if rows:
        for key, value in rows:
            print(f"{str(key):<24}: {format_value(value)}")
    print("=" * width)


def print_table(title, headers, rows, width=78):
    print("=" * width)
    print(title.upper())
    print("-" * width)
    if not rows:
        print("(no data)")
        print("=" * width)
        return

    table_rows = [[format_value(cell) for cell in row] for row in rows]
    header_cells = [str(header) for header in headers]
    widths = []
    for idx, header in enumerate(header_cells):
        max_cell = max(len(row[idx]) for row in table_rows) if table_rows else 0
        widths.append(max(len(header), max_cell, 8))

    header_line = " | ".join(header_cells[idx].ljust(widths[idx]) for idx in range(len(header_cells)))
    print(header_line)
    print("-" * len(header_line))
    for row in table_rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(row))))
    print("=" * width)


def make_log_context(args, setting=None, round_idx=None, sub_round_idx=None):
    return {
        "model": getattr(args, "model", None),
        "dataset": getattr(args, "dataset", None),
        "setting": getattr(args, "setting", None),
        "experiment_mode": getattr(setting, "experiment_mode", getattr(args, "experiment_mode", None)),
        "split_type": getattr(setting, "split_type", getattr(args, "split_type", None)),
        "label_used": getattr(args, "label_used", None),
        "round": None if round_idx is None else round_idx + 1,
        "sub_round": None if sub_round_idx is None else sub_round_idx + 1,
    }


def _shape_of(value):
    if isinstance(value, dict):
        return {key: _shape_of(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and not hasattr(value, "shape"):
        return [_shape_of(item) for item in value]
    return getattr(value, "shape", None)


def log_split_info(train_indexes=None, test_indexes=None, val_indexes=None, train_data=None, val_data=None,
                   test_data=None, test_sub_label=None, context=None):
    rows = []
    if context:
        for key in ("round", "sub_round"):
            if context.get(key) is not None:
                rows.append((key, context[key]))
    rows.extend([
        ("train indexes", train_indexes),
        ("val indexes", val_indexes),
        ("test indexes", test_indexes),
    ])
    if train_data is not None:
        rows.append(("train data shape", _shape_of(train_data)))
    if val_data is not None:
        rows.append(("val data shape", _shape_of(val_data)))
    if test_data is not None:
        rows.append(("test data shape", _shape_of(test_data)))
    if test_sub_label is not None:
        rows.append(("test subject labels", _shape_of(test_sub_label)))
    print_section("Split", rows)


class ExperimentLogger:
    def __init__(self, output_dir=None, log_context=None, history_filename="training_history.csv"):
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.context = self._clean_context(log_context or {})
        self.history_path = None
        if self.output_dir is not None:
            self.history_path = self.output_dir / history_filename

    def log_epoch(self, epoch, total_epochs, lr, train_metrics, val_metrics, best_metrics=None,
                  improved_metrics=None, stage="train"):
        best_metrics = best_metrics or {}
        improved_metrics = set(improved_metrics or [])
        train_metrics = train_metrics or {}
        val_metrics = val_metrics or {}

        metric_names = metric_order(set(train_metrics) | set(val_metrics) | set(best_metrics))
        rows = []
        for name in metric_names:
            status = "new best" if name in improved_metrics else ""
            rows.append([
                name,
                train_metrics.get(name),
                val_metrics.get(name),
                best_metrics.get(name),
                status,
            ])

        title = f"{stage} epoch {epoch}/{total_epochs}  lr={format_lr(lr)}"
        print_table(title, ["metric", "train", "val", "best_val", "status"], rows)

        history_row = dict(self.context)
        history_row.update({
            "stage": stage,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "lr": format_lr(lr),
            "improved_metrics": ",".join(metric_order(improved_metrics)),
        })
        history_row.update(self._prefix_metrics("train", train_metrics))
        history_row.update(self._prefix_metrics("val", val_metrics))
        history_row.update(self._prefix_metrics("best_val", best_metrics))
        self.append_history(history_row)

    def log_final(self, test_metrics, best_metrics=None, metrics=None, stage="test"):
        best_metrics = best_metrics or {}
        test_metrics = test_metrics or {}
        metric_names = set(test_metrics) | set(best_metrics)
        if metrics:
            metric_names.update(metrics)
        rows = []
        for name in metric_order(metric_names):
            rows.append([name, best_metrics.get(name), test_metrics.get(name)])
        print_table("Final Result", ["metric", "best_val", stage], rows)

        history_row = dict(self.context)
        history_row.update({
            "stage": f"final_{stage}",
            "epoch": "",
            "total_epochs": "",
            "lr": "",
            "improved_metrics": "",
        })
        history_row.update(self._prefix_metrics(stage, test_metrics))
        history_row.update(self._prefix_metrics("best_val", best_metrics))
        self.append_history(history_row)

    def append_history(self, row):
        if self.history_path is None:
            return
        os.makedirs(self.history_path.parent, exist_ok=True)
        clean_row = {key: self._stringify(value) for key, value in row.items()}

        if not self.history_path.exists():
            with open(self.history_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(clean_row.keys()))
                writer.writeheader()
                writer.writerow(clean_row)
            return

        with open(self.history_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        new_fields = [key for key in clean_row if key not in fieldnames]
        if new_fields:
            fieldnames.extend(new_fields)
            rows.append(clean_row)
            with open(self.history_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return

        with open(self.history_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow(clean_row)

    @staticmethod
    def _prefix_metrics(prefix, metrics):
        return {f"{prefix}_{key}": value for key, value in (metrics or {}).items()}

    @classmethod
    def _clean_context(cls, context):
        return {key: cls._stringify(value) for key, value in context.items() if value is not None}

    @staticmethod
    def _stringify(value):
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        return format_value(value)
