#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import atacworks.dl4atac.metrics as metrics

import pytest

import torch

test_cases = [
    # CorrCoef
    (
        metrics.CorrCoef(),
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32),
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32),
        torch.tensor(1.)
    ),
    (
        metrics.CorrCoef(),
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32),
        torch.tensor([-1, -2, -3, -4, -5, -6, -7, -8], dtype=torch.float32),
        torch.tensor(-1.)
    ),
    (
        metrics.CorrCoef(),
        torch.tensor([1, 4, 9, 16, 25, 36, 49, 64], dtype=torch.float32),
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32),
        torch.tensor(0.9762)
    ),
    # Recall
    (
        metrics.Recall(0.49),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1. / 3.)
    ),
    (
        metrics.Recall(0.0),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1.)
    ),
    (
        metrics.Recall(0.99),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(0.)
    ),
    # Specificity
    (
        metrics.Specificity(0.49),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1. / 3.)
    ),
    (
        metrics.Specificity(0.0),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(0.)
    ),
    (
        metrics.Specificity(0.99),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1.)
    ),
    # Precision
    (
        metrics.Precision(0.49),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1. / 3.)
    ),
    (
        metrics.Precision(0.0),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1. / 2.)
    ),
    (
        metrics.Precision(0.99),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1.)
    ),
    # NPV
    (
        metrics.NPV(0.49),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1. / 3.)
    ),
    (
        metrics.NPV(0.0),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1.)
    ),
    (
        metrics.NPV(0.99),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1. / 2.)
    ),
    # Accuracy
    (
        metrics.Accuracy(0.49),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1. / 3.)
    ),
    (
        metrics.Accuracy(0.49),
        torch.tensor([[0.4, 0.5, 0.6], [0.7, 0.2, 0.3]], dtype=torch.float32),
        torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.float32),
        torch.tensor(1. / 3.)
    ),
    (
        metrics.Accuracy(0.0),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1. / 2.)
    ),
    (
        metrics.Accuracy(0.99),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.2, 0.3], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.tensor(1. / 2.)
    ),
]


@pytest.mark.gpu
@pytest.mark.parametrize("metrics_class, x, y, correct_res", test_cases)
def test_metrics_gpu(metrics_class, x, y, correct_res):
    """
    Tests to run metrics calculations on GPU.
    """
    x = x.cuda()
    y = y.cuda()
    res = metrics_class(x, y)
    res = res.cpu()
    assert (torch.allclose(res, correct_res, atol=1e-4, rtol=1e-4))


@pytest.mark.cpu
@pytest.mark.parametrize("metrics_class, x, y, correct_res", test_cases)
def test_metrics_cpu(metrics_class, x, y, correct_res):
    """
    Tests to run metrics calculations on CPU.
    """
    res = metrics_class(x, y)
    assert (torch.allclose(res, correct_res, atol=1e-4, rtol=1e-4))
