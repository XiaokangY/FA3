from typing import Union, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import eagerpy as ep
import numpy as np
from scipy.stats import norm
import torch.nn.functional as F
import torch
import librosa
import torchvision.transforms as transforms
from foolbox.devutils import flatten
from foolbox.devutils import atleast_kd
from foolbox.types import Bounds
from foolbox.models.base import Model
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.distances import l1, l2, linf
from foolbox.attacks.base import FixedEpsilonAttack
from foolbox.attacks.base import T
from foolbox.attacks.base import get_criterion
from foolbox.attacks.base import raise_if_kwargs
from foolbox.attacks.base import verify_input_bounds
from typing import Optional
from foolbox.attacks.gradient_descent_base import AdamOptimizer, GDOptimizer, Optimizer
from foolbox.attacks.gradient_descent_base import LinfBaseGradientDescent
from foolbox.attacks.deepfool import LinfDeepFoolAttack
from typing import Union, Optional, Tuple, Any, Callable
from tqdm import tqdm



class FA3_PGD(LinfBaseGradientDescent):
    def __init__(
            self,
            *,
            rel_stepsize: float = 0.01 / 0.3,
            abs_stepsize: Optional[float] = None,
            steps: int = 40,
            random_start: bool = True,
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )

    def run(
            self,
            model: Model,
            inputs: T,
            criterion: Union[Misclassification, TargetedMisclassification, T],
            *,
            epsilon: float,
            lamb: float = None,
            **kwargs: Any,

    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        verify_input_bounds(x0, model)

        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes = criterion_.labels
        elif hasattr(criterion_, "target_classes"):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes  # type: ignore
        else:
            raise ValueError("unsupported criterion")

        loss_fn = self.get_loss_fn(model, classes)
        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        optimizer = self.get_optimizer(x0, stepsize)

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model.bounds)
        else:
            x = x0
        x0 = x0.raw
        # 频率权重数组 weights
        freqs = librosa.mel_frequencies(n_mels=128)
        weights = librosa.A_weighting(freqs)

        normalized_weights = np.zeros_like(weights)
        normalized_weights[0:39] = np.interp(weights[0:39], (np.min(weights[0:39]), np.max(weights[0:39])),
                                             (0.08, 0.028))
        normalized_weights[38:105] = np.interp(weights[38:105], (np.min(weights[38:105]), np.max(weights[38:105])),
                                               (0.03, 0.028))
        normalized_weights[104:] = np.interp(weights[104:], (np.min(weights[104:]), np.max(weights[104:])),
                                             (normalized_weights[19], 0.03))

        # 将权重线性映射到目标范围
        # normalized_weights = np.interp(weights, (np.min(weights), np.max(weights)), (0.06,0.02))

        normalized_weights = torch.from_numpy(normalized_weights / 40)
        normalized_weights = normalized_weights.unsqueeze(1)
        normalized_weights = normalized_weights.to('cuda:0')

        n = 128
        for i in range(0, 128 - n + 1, n):
            print(i)
            for _ in tqdm(range(self.steps)):
                _, gradients = self.value_and_grad(loss_fn, x)
                gradients = self.normalize(gradients, x=x, bounds=model.bounds)
                x = x.raw
                gradients = gradients.raw
                x[:, :, i:i + n, :] = x[:, :, i:i + n, :] + gradient_step_sign * gradients[:, :, i:i + n,
                                                                                 :] * normalized_weights[i:i + n]

                for j in range(i, i + n):
                    x[:, :, j, :] = x0[:, :, j, :] + (x - x0).clip(-normalized_weights[j] * 40.0,
                                                                   normalized_weights[j] * 40.0)[:, :, j, :]

                x = ep.astensor(x)
                x = ep.clip(x, *model.bounds)
        x0 = ep.astensor(x0)
        return restore_type(x)

class MIM(LinfBaseGradientDescent):
    def __init__(
            self,
            *,
            rel_stepsize: float = 0.01 / 0.3,
            abs_stepsize: Optional[float] = None,
            steps: int = 40,
            random_start: bool = True,
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )

    def run(
            self,
            model: Model,
            inputs: T,
            criterion: Union[Misclassification, TargetedMisclassification, T],
            *,
            epsilon: float,
            **kwargs: Any,

    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        verify_input_bounds(x0, model)

        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes = criterion_.labels
        elif hasattr(criterion_, "target_classes"):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes  # type: ignore
        else:
            raise ValueError("unsupported criterion")
        loss_fn = self.get_loss_fn(model, classes)
        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        optimizer = self.get_optimizer(x0, stepsize)

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model.bounds)
        else:
            x = x0

        momentum = ep.zeros_like(x)
        for i in range(self.steps):
            _, gradients = self.value_and_grad(loss_fn, x)
            gradients = self.normalize(gradients, x=x, bounds=model.bounds)
            momentum = gradients + 0.1 * momentum
            x = x + gradient_step_sign * optimizer(momentum)
            x = self.project(x, x0, epsilon)
            x = ep.clip(x, *model.bounds)

        return restore_type(x)

class FA3_FGSM(LinfBaseGradientDescent):
    def __init__(
            self,
            *,
            rel_stepsize: float = 1,
            abs_stepsize: Optional[float] = None,
            steps: int = 1,
            random_start: bool = True,
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )

    def run(
            self,
            model: Model,
            inputs: T,
            criterion: Union[Misclassification, TargetedMisclassification, T],
            *,
            epsilon: float,
            **kwargs: Any,

    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        verify_input_bounds(x0, model)

        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes = criterion_.labels
        elif hasattr(criterion_, "target_classes"):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes  # type: ignore
        else:
            raise ValueError("unsupported criterion")
        loss_fn = self.get_loss_fn(model, classes)
        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        optimizer = self.get_optimizer(x0, stepsize)

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model.bounds)
        else:
            x = x0

        x0 = x0.raw
        # 频率权重数组 weights
        freqs = librosa.mel_frequencies(n_mels=128)
        weights = librosa.A_weighting(freqs)

        normalized_weights = np.zeros_like(weights)
        normalized_weights[0:39] = np.interp(weights[0:39], (np.min(weights[0:39]), np.max(weights[0:39])),
                                             (0.08, 0.03))
        normalized_weights[38:105] = np.interp(weights[38:105], (np.min(weights[38:105]), np.max(weights[38:105])),
                                               (0.03, 0.028))
        normalized_weights[104:] = np.interp(weights[104:], (np.min(weights[104:]), np.max(weights[104:])),
                                             (normalized_weights[19], 0.03))

        # 将权重线性映射到目标范围
        # normalized_weights = np.interp(weights, (np.min(weights), np.max(weights)), (0.06,0.02))

        normalized_weights = torch.from_numpy(normalized_weights / 1)
        normalized_weights = normalized_weights.unsqueeze(1)
        normalized_weights = normalized_weights.to('cuda:0')

        n = 128
        for i in range(0, 128 - n + 1, n):
            print(i)
            for _ in tqdm(range(self.steps)):
                _, gradients = self.value_and_grad(loss_fn, x)
                gradients = self.normalize(gradients, x=x, bounds=model.bounds)
                x = x.raw
                gradients = gradients.raw
                x[:, :, i:i + n, :] = x[:, :, i:i + n, :] + gradient_step_sign * gradients[:, :, i:i + n,
                                                                                 :] * normalized_weights[i:i + n]

                for j in range(i, i + n):
                    x[:, :, j, :] = x0[:, :, j, :] + (x - x0).clip(-normalized_weights[j] * 1.0,
                                                                   normalized_weights[j] * 1.0)[:, :, j, :]

                x = ep.astensor(x)
                x = ep.clip(x, *model.bounds)
        x0 = ep.astensor(x0)
        return restore_type(x)

class FA3_MIM(LinfBaseGradientDescent):
    def __init__(
            self,
            *,
            rel_stepsize: float = 0.01 / 0.3,
            abs_stepsize: Optional[float] = None,
            steps: int = 40,
            random_start: bool = True,
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )

    def run(
            self,
            model: Model,
            inputs: T,
            criterion: Union[Misclassification, TargetedMisclassification, T],
            *,
            epsilon: float,
            **kwargs: Any,

    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        verify_input_bounds(x0, model)

        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes = criterion_.labels
        elif hasattr(criterion_, "target_classes"):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes  # type: ignore
        else:
            raise ValueError("unsupported criterion")
        loss_fn = self.get_loss_fn(model, classes)
        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        optimizer = self.get_optimizer(x0, stepsize)

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model.bounds)
        else:
            x = x0

        x0 = x0.raw
        # 频率权重数组 weights
        freqs = librosa.mel_frequencies(n_mels=128)
        weights = librosa.A_weighting(freqs)

        normalized_weights = np.zeros_like(weights)
        normalized_weights[0:39] = np.interp(weights[0:39], (np.min(weights[0:39]), np.max(weights[0:39])),
                                             (0.08, 0.028))
        normalized_weights[38:105] = np.interp(weights[38:105], (np.min(weights[38:105]), np.max(weights[38:105])),
                                               (0.03, 0.028))
        normalized_weights[104:] = np.interp(weights[104:], (np.min(weights[104:]), np.max(weights[104:])),
                                             (normalized_weights[19], 0.03))

        # 将权重线性映射到目标范围
        # normalized_weights = np.interp(weights, (np.min(weights), np.max(weights)), (0.06,0.02))

        normalized_weights = torch.from_numpy(normalized_weights / 40)
        normalized_weights = normalized_weights.unsqueeze(1)
        normalized_weights = normalized_weights.to('cuda:0')

        n = 128
        for i in range(0, 128 - n + 1, n):
            momentum = ep.zeros_like(x)
            for _ in tqdm(range(self.steps)):
                _, gradients = self.value_and_grad(loss_fn, x)
                gradients = self.normalize(gradients, x=x, bounds=model.bounds)
                momentum = gradients + 0.1 * momentum
                x = x.raw
                momentum = momentum.raw
                x[:, :, i:i + n, :] = x[:, :, i:i + n, :] + gradient_step_sign * momentum[:, :, i:i + n,
                                                                             :] * normalized_weights[i:i + n]

                for j in range(i, i + n):
                    x[:, :, j, :] = x0[:, :, j, :] + (x - x0).clip(-normalized_weights[j] * 40.0,
                                                               normalized_weights[j] * 40.0)[:, :, j, :]
                momentum=ep.astensor(momentum)
                x = ep.astensor(x)
                x = ep.clip(x, *model.bounds)
        x0 = ep.astensor(x0)
        return restore_type(x)