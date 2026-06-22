from collections import defaultdict

import torch
from pytorch_lightning.callbacks import Callback, TQDMProgressBar


class ScientificProgressBar(TQDMProgressBar):
    def __init__(
        self,
        refresh_rate: int = 1,
        process_position: int = 0,
        metric_update_interval: int = 10,
    ):
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)
        self.metric_update_interval = metric_update_interval
        self._last_metrics = {}
        self._metrics_accumulator = defaultdict(float)
        self._metrics_count = defaultdict(int)
        self._steps_since_update = 0

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self._reset_accumulator()
        self._last_metrics = {}

    def _reset_accumulator(self):
        self._metrics_accumulator.clear()
        self._metrics_count.clear()
        self._steps_since_update = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Accumulate metrics
        # Use super().get_metrics() to fetch current step's raw metrics
        # (includes v_num which we ignore)
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)

        for k, v in items.items():
            val = None
            if isinstance(v, (float, int)):
                val = float(v)
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                val = v.item()

            if val is not None:
                self._metrics_accumulator[k] += val
                self._metrics_count[k] += 1

        self._steps_since_update += 1

        if self._steps_since_update >= self.metric_update_interval:
            # Compute averages
            new_metrics = {}
            for k, count in self._metrics_count.items():
                if count > 0:
                    new_metrics[k] = self._metrics_accumulator[k] / count

            # Format and update cache
            self._last_metrics = self._format_items(new_metrics)
            self._reset_accumulator()

        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def get_metrics(self, trainer, model):
        if trainer.training:
            return self._last_metrics
        else:
            # For validation/test, display current metrics directly
            items = super().get_metrics(trainer, model)
            items.pop("v_num", None)
            return self._format_items(items)

    def _format_items(self, items):
        formatted = {}
        for k, v in items.items():
            if isinstance(v, (float, int)):
                formatted[k] = f"{v:.3e}"
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                formatted[k] = f"{v.item():.3e}"
            else:
                formatted[k] = v
        return formatted


class MetricAliasCallback(Callback):
    def __init__(self, alias_map):
        super().__init__()
        self.alias_map = alias_map

    def on_train_epoch_end(self, trainer, pl_module):
        self._alias_metrics(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._alias_metrics(trainer)

    def _alias_metrics(self, trainer):
        metrics = trainer.callback_metrics
        for src, dest in self.alias_map.items():
            if src in metrics:
                metrics[dest] = metrics[src]
