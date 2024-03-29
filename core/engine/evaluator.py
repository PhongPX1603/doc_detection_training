import time
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import prepare_device


class Evaluator(nn.Module):
    def __init__(
        self,
        data: nn.Module,
        model: nn.Module,
        metric: Callable,
        predictor: Callable
    ):
        super(Evaluator, self).__init__()
        self.data = data
        self.model = model
        self.metric = metric
        self.predictor = predictor

    def eval_epoch(self, evaluator_name: str, dataloader: nn.Module) -> Dict[str, float]:
        self.model.eval()
        self.metric.started(evaluator_name)
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                params = [param for param in batch]
                samples = torch.stack([image.to(self.device) for image in params[0]], dim=0)
                targets = [{k: v.to(self.device) for k, v in target.items() if not isinstance(v, list)} for target in params[1]]
                preds, cls_preds, reg_preds, anchors = self.model(samples)
                
                if self.predictor is not None:
                    self.predictor.update([preds, params[2]])
                
                _ = self.metric.iteration_completed(output=[preds, cls_preds, reg_preds, anchors, targets, params[2]])

        return self.metric.epoch_completed()

    def __call__(self, checkpoint_path: Optional[str] = None, num_gpus: int = 0):
        # Load weight
        if checkpoint_path is None:
            raise ValueError('No checkpoint to load.')

        # Set Device for Model: prepare for (multi-device) GPU training
        self.device, gpu_indices = prepare_device(num_gpus)
        self.model = self.model.to(self.device)
        if len(gpu_indices) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_indices)

        # Load weight
        state_dict = torch.load(f=checkpoint_path, map_location=self.device)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state_dict=state_dict)
        else:
            self.model.load_state_dict(state_dict=state_dict)

        # Start to evaluate
        print(f'{time.asctime()} - STARTED')
        metrics = self.eval_epoch(evaluator_name='test', dataloader=self.data)
        messages = [f'\n* {metric_name}:\n{metric_value}\n' for metric_name, metric_value in metrics.items()]
        print(f"[INFO] {''.join(messages)}")
        print(f'{time.asctime()} - COMPLETED')
