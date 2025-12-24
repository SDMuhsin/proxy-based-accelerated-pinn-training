"""
Logging and evaluation utilities
"""

from sklearn import metrics
import numpy as np
import logging


def get_logger(log_name='log.txt'):
    """Create a logger instance with file and console handlers"""
    logger = logging.getLogger('mylogger')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - function:%(funcName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_name is not None:
        handler = logging.FileHandler(log_name)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_metrix(true_label, pred_label):
    """Calculate evaluation metrics: MAE, MAPE, MSE, RMSE"""
    MAE = metrics.mean_absolute_error(true_label, pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label, pred_label)
    MSE = metrics.mean_squared_error(true_label, pred_label)
    RMSE = np.sqrt(metrics.mean_squared_error(true_label, pred_label))

    return [MAE, MAPE, MSE, RMSE]


def write_to_txt(txt_name, txt):
    """Append text to a file"""
    with open(txt_name, 'a') as f:
        f.write(txt)
        f.write('\n')
