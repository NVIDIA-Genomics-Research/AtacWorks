#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import torch
import torch.distributed as dist
import torch.nn.functional as F
import sklearn.metrics
from collections import Iterable


class Metric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = torch.tensor(0.).cuda() if torch.cuda.is_available() else torch.tensor(0.)

    def get(self):
        return self.val

    def __str__(self):
        return "{}:{:7.4f}".format(type(self).__name__.lower(), self.val)

    def __call__(self, x, y):
        raise NotImplementedError("Abstract class method called.")

    def better_than(self, metric):
        raise NotImplementedError("Abstract class method called.")

    @staticmethod
    def convert_to_tensor(x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return x

    @staticmethod
    def convert_to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return x


class BCE(Metric):
    def __init__(self):
        super(BCE, self).__init__()

    def __call__(self, x, y):
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        self.val = F.binary_cross_entropy(x, y)
        return self.val

    def better_than(self, metric):
        if not metric: 
            return True
        return self.get() < metric.get()


class MSE(Metric):
    def __init__(self):
        super(MSE, self).__init__()

    def __call__(self, x, y):
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        self.val = F.mse_loss(x, y)
        return self.val

    def better_than(self, metric):
        if not metric: 
            return True
        return self.get() < metric.get()


class CorrCoef(Metric):
    def __init__(self):
        super(CorrCoef, self).__init__()

    def __call__(self, x, y):
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        x_norm = x - torch.mean(x)
        y_norm = y - torch.mean(y)

        self.val = torch.sum(x_norm * y_norm) / (torch.sqrt(torch.sum(x_norm**2))
                                             * torch.sqrt(torch.sum(y_norm**2)))
        return self.val

    def better_than(self, metric):
        if not metric: 
            return True
        return self.get() > metric.get()


class Recall(Metric):
    def __init__(self, threshold):
        super(Recall, self).__init__()
        self.threshold = threshold

    def __call__(self, x, y):
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        num_peaks_label = torch.sum(y == 1).type(torch.float32)
        num_peaks_pred = torch.sum(x[y == 1] > self.threshold).type(torch.float32)

        if num_peaks_label:
            self.val = num_peaks_pred / num_peaks_label
        else:
            self.val = torch.tensor(1.).cuda() if x.is_cuda else torch.tensor(1.)
        return self.val

    def better_than(self, metric):
        if not metric: 
            return True
        return self.get() > metric.get()


class Specificity(Metric):
    def __init__(self, threshold):
        super(Specificity, self).__init__()
        self.threshold = threshold

    def __call__(self, x, y):
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        num_nonpeaks_label = torch.sum(y == 0).type(torch.float32)
        num_nonpeaks_pred = torch.sum((x[y == 0] <= self.threshold)).type(torch.float32)

        if num_nonpeaks_label:
            self.val = num_nonpeaks_pred / num_nonpeaks_label
        else:
            self.val = torch.tensor(1.).cuda() if x.is_cuda else torch.tensor(1.)
        return self.val

    def better_than(self, metric):
        if not metric: 
            return True
        return self.get() > metric.get()


class Precision(Metric):
    def __init__(self, threshold):
        super(Precision, self).__init__()
        self.threshold = threshold

    def __call__(self, x, y):
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        num_peak_correct = torch.sum((y[x > self.threshold] == 1)).type(torch.float32)
        num_peak_pred = torch.sum(x > self.threshold).type(torch.float32)

        if num_peak_pred:
            self.val = num_peak_correct / num_peak_pred
        else:
            self.val = torch.tensor(1.).cuda() if x.is_cuda else torch.tensor(1.)
        return self.val

    def better_than(self, metric):
        if not metric: 
            return True
        return self.get() > metric.get()


class NPV(Metric):
    def __init__(self, threshold):
        super(NPV, self).__init__()
        self.threshold = threshold

    def __call__(self, x, y):
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        num_nonpeak_correct = torch.sum((y[x <= self.threshold] == 0)).type(torch.float32)
        num_nonpeak_pred = torch.sum(x <= self.threshold).type(torch.float32)

        if num_nonpeak_pred:
            self.val = num_nonpeak_correct / num_nonpeak_pred
        else:
            self.val = torch.tensor(1.).cuda() if x.is_cuda else torch.tensor(1.)
        return self.val

    def better_than(self, metric):
        if not metric:
            return True
        return self.get() > metric.get()


class Accuracy(Metric):
    def __init__(self, threshold):
        super(Accuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, x, y):
        x = Metric.convert_to_tensor(x)
        y = Metric.convert_to_tensor(y)
        tp = torch.sum((x[y==1] > self.threshold)).type(torch.float32)
        tn = torch.sum(x[y==0] <= self.threshold).type(torch.float32)
        count = x.nelement()
        correct = tp + tn
        self.val = correct/count
        return self.val

    def better_than(self, metric):
        if not metric: 
            return True
        return self.get() > metric.get()


class AUROC(Metric):
    def __init__(self):
        super(AUROC, self).__init__()

    def __call__(self, x, y):
        x = Metric.convert_to_numpy(x)
        y = Metric.convert_to_numpy(y)
        x = x.flatten()
        y = y.flatten()
        self.val = sklearn.metrics.roc_auc_score(y, x)
        return self.val

    def better_than(self, metric):
        if not metric:
            return True
        return self.get() > metric.get()


class AUPRC(Metric):
    def __init__(self):
        super(AUPRC, self).__init__()

    def __call__(self, x, y):
        x = Metric.convert_to_numpy(x)
        y = Metric.convert_to_numpy(y)
        x = x.flatten()
        y = y.flatten()
        self.val = sklearn.metrics.average_precision_score(y, x)
        return self.val

    def better_than(self, metric):
        if not metric:
            return True
        return self.get() > metric.get()
