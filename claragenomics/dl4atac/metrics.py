#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Metrics module."""

import numpy as np

from scipy.stats import rankdata

import sklearn.metrics

import torch

import torch.nn.functional as F


class Metric(object):
    """Base metrics class."""

    def __init__(self):
        """Initialize class."""
        self.reset()

    def reset(self):
        """Reset value of metric."""
        self.val = torch.tensor(
            0.).cuda() if torch.cuda.is_available() else torch.tensor(0.)

    def get(self):
        """Get metrics value."""
        return self.val

    def __str__(self):
        """Serialize metric into a string."""
        return "{}:{:7.4f}".format(type(self).__name__.lower(), self.val)

    def __call__(self, x, y):
        """Call the metric class as a function.

        Args:
            x: Vector of predicted values.
            y: Vector of target values.

        Raise:
            Not implemented error

        """
        raise NotImplementedError("Abstract class method called.")

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.
            Not implemented error if not implemented.

        """
        raise NotImplementedError("Abstract class method called.")

    @staticmethod
    def convert_to_tensor(x):
        """Convert value to torch tensor.

        Args:
            x: Input vector as numpy array.

        Return:
            x: Torch tensor encoded value.

        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return x

    @staticmethod
    def convert_to_numpy(x):
        """Convert value to numpy array.

        Args:
            x: Input vector as torch tensor.

        Return:
            x: Numpy encoded value.

        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return x


class BCE(Metric):
    """Binary cross entropy."""

    def __init__(self):
        """Initialize."""
        super(BCE, self).__init__()

    def __call__(self, x, y):
        """Calculate binary cross entropy.

        Args:
            x: Prediction vector
            y: Target vector

        Return:
            value: Binary cross entropy between prediction and target.

        """
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        self.val = F.binary_cross_entropy(x, y)
        return self.val

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.

        """
        if not metric:
            return True
        return self.get() < metric.get()


class MSE(Metric):
    """Mean squared error."""

    def __init__(self):
        """Initialize."""
        super(MSE, self).__init__()

    def __call__(self, x, y):
        """Calculate mean square error.

        Args:
            x : Prediction vector
            y : Target vector

        Return:
            Mean squared error.

        """
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        self.val = F.mse_loss(x, y)
        return self.val

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.

        """
        if not metric:
            return True
        return self.get() < metric.get()


class CorrCoef(Metric):
    """Correlated coefficient."""

    def __init__(self):
        """Initialize."""
        super(CorrCoef, self).__init__()

    def __call__(self, x, y, eps=0):
        """Calculate correlated coefficient metric.

        Args:
            x: Prediction vector
            y: Target vector
            eps: Epsilon value to add to the loss, to prevent division by 0.

        Return:
            value: Binary cross entropy between prediction and target.

        """
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        xm = x.mean()
        ym = y.mean()
        x = x - xm
        y = y - ym
        self.val = torch.sum(x * y) / (
            torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(
                torch.sum(y ** 2)) + eps)
        return self.val

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.

        """
        if not metric:
            return True
        return self.get() > metric.get()


class Recall(Metric):
    """Recall."""

    def __init__(self, threshold):
        """Initialize."""
        super(Recall, self).__init__()
        self.threshold = threshold

    def __call__(self, x, y):
        """Calculate recall.

        Args:
            x: Prediction vector
            y: Target vector

        Return:
            value: Binary cross entropy between prediction and target.

        """
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        num_peaks_label = torch.sum(y == 1).type(torch.float32)
        num_peaks_pred = torch.sum(x[y == 1] > self.threshold).type(
            torch.float32)

        if num_peaks_label:
            self.val = num_peaks_pred / num_peaks_label
        else:
            self.val = torch.tensor(1.).cuda() if x.is_cuda else torch.tensor(
                1.)
        return self.val

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.

        """
        if not metric:
            return True
        return self.get() > metric.get()


class Specificity(Metric):
    """Specificity."""

    def __init__(self, threshold):
        """Initialize."""
        super(Specificity, self).__init__()
        self.threshold = threshold

    def __call__(self, x, y):
        """Calculate specificity.

        Args:
            x: Prediction vector
            y: Target vector

        Return:
            value: Binary cross entropy between prediction and target.

        """
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        num_nonpeaks_label = torch.sum(y == 0).type(torch.float32)
        num_nonpeaks_pred = torch.sum((x[y == 0] <= self.threshold)).type(
            torch.float32)

        if num_nonpeaks_label:
            self.val = num_nonpeaks_pred / num_nonpeaks_label
        else:
            self.val = torch.tensor(1.).cuda() if x.is_cuda else torch.tensor(
                1.)
        return self.val

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.

        """
        if not metric:
            return True
        return self.get() > metric.get()


class Precision(Metric):
    """Precision."""

    def __init__(self, threshold):
        """Initialize."""
        super(Precision, self).__init__()
        self.threshold = threshold

    def __call__(self, x, y):
        """Calculate precision.

        Args:
            x: Prediction vector
            y: Target vector

        Return:
            value: Binary cross entropy between prediction and target.

        """
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        num_peak_correct = torch.sum((y[x > self.threshold] == 1)).type(
            torch.float32)
        num_peak_pred = torch.sum(x > self.threshold).type(torch.float32)

        if num_peak_pred:
            self.val = num_peak_correct / num_peak_pred
        else:
            self.val = torch.tensor(1.).cuda() if x.is_cuda else torch.tensor(
                1.)
        return self.val

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.

        """
        if not metric:
            return True
        return self.get() > metric.get()


class F1(Metric):
    """F1."""

    def __init__(self, threshold):
        """Initialize."""
        super(F1, self).__init__()
        self.threshold = threshold

    def __call__(self, x, y):
        """Calculate F1 metric.

        Args:
            x: Prediction vector
            y: Target vector

        Return:
            value: Binary cross entropy between prediction and target.

        """
        r = Recall(self.threshold)
        p = Precision(self.threshold)
        rec = r(x, y)
        prec = p(x, y)
        self.val = 2 * rec * prec / (rec + prec)
        return self.val

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.

        """
        if not metric:
            return True
        return self.get() > metric.get()


class NPV(Metric):
    """Negative predictd value."""

    def __init__(self, threshold):
        """Initialize."""
        super(NPV, self).__init__()
        self.threshold = threshold

    def __call__(self, x, y):
        """Calculate negative predicted value.

        Args:
            x: Prediction vector
            y: Target vector

        Return:
            value: Binary cross entropy between prediction and target.

        """
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        num_nonpeak_correct = torch.sum((y[x <= self.threshold] == 0)).type(
            torch.float32)
        num_nonpeak_pred = torch.sum(x <= self.threshold).type(torch.float32)

        if num_nonpeak_pred:
            self.val = num_nonpeak_correct / num_nonpeak_pred
        else:
            self.val = torch.tensor(1.).cuda() if x.is_cuda else torch.tensor(
                1.)
        return self.val

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.

        """
        if not metric:
            return True
        return self.get() > metric.get()


class Accuracy(Metric):
    """Accuracy."""

    def __init__(self, threshold):
        """Initialize."""
        super(Accuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, x, y):
        """Calculate accuracy.

        Args:
            x: Prediction vector
            y: Target vector

        Return:
            value: Binary cross entropy between prediction and target.

        """
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        tp = torch.sum((x[y == 1] > self.threshold)).type(torch.float32)
        tn = torch.sum(x[y == 0] <= self.threshold).type(torch.float32)
        count = x.nelement()
        correct = tp + tn
        self.val = correct / count
        return self.val

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.

        """
        if not metric:
            return True
        return self.get() > metric.get()


class AUROC(Metric):
    """Area under Receiver-Operating Characteristics curve."""

    def __init__(self):
        """Initialize."""
        super(AUROC, self).__init__()

    def __call__(self, x, y):
        """Calculate area under Receiver Operating Characteristics curve.

        Args:
            x: Prediction vector
            y: Target vector

        Return:
            value: Binary cross entropy between prediction and target.

        """
        x = Metric.convert_to_numpy(x)
        y = Metric.convert_to_numpy(y)
        x = x.flatten()
        y = y.flatten()
        self.val = sklearn.metrics.roc_auc_score(y, x)
        return self.val

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.

        """
        if not metric:
            return True
        return self.get() > metric.get()


class AUPRC(Metric):
    """Area under Precision-Recall curve."""

    def __init__(self):
        """Initialize."""
        super(AUPRC, self).__init__()

    def __call__(self, x, y):
        """Calculate area under Precision-Recall curve.

        Args:
            x: Prediction vector
            y: Target vector

        Return:
            value: Binary cross entropy between prediction and target.

        """
        x = Metric.convert_to_numpy(x)
        y = Metric.convert_to_numpy(y)
        x = x.flatten()
        y = y.flatten()
        self.val = sklearn.metrics.average_precision_score(y, x)
        return self.val

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.

        """
        if not metric:
            return True
        return self.get() > metric.get()


class SpearmanCorrCoef(Metric):
    """Spearman correlated coefficient."""

    def __init__(self):
        """Initialize."""
        super(SpearmanCorrCoef, self).__init__()

    def __call__(self, x, y):
        """Calculate Spearman correlated coefficient.

        Args:
            x: Prediction vector
            y: Target vector

        Return:
            value: Binary cross entropy between prediction and target.

        """
        x = Metric.convert_to_numpy(x)
        y = Metric.convert_to_numpy(y)
        x = np.apply_along_axis(rankdata, 0, x)
        y = np.apply_along_axis(rankdata, 0, y)
        xm = x.mean()
        ym = y.mean()
        np.subtract(x, xm, out=x)
        np.subtract(y, ym, out=y)
        self.val = np.sum(x * y) / (
            np.sqrt(np.sum(x ** 2)) * np.sqrt(
                np.sum(y ** 2)))
        return self.val

    def better_than(self, metric):
        """Check if current metric is better than given metric.

        Args:
            metric: Another instance of metric.

        Return:
            True : if given metric is greater than current metric.
            False : if given metric is lesser than current metric.

        """
        if not metric:
            return True
        return self.get() > metric.get()
