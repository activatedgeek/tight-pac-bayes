import functools
from datetime import datetime
import pandas as pd


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params {total:4,d}")


def save_metrics(bound_metrics, quant_epochs, levels):
    bound_metrics['epochs'] = quant_epochs
    bound_metrics['levels'] = levels
    df = pd.DataFrame([bound_metrics])
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H%M_%S")
    file_path = f"./res30_{levels}_" + time_stamp + ".csv"
    df.to_csv(file_path, index=False)
