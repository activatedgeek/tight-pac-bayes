import timm
import pandas as pd
from timm.models import create_model

# choices = [m for m in timm.list_models() if m.find("mobile") >= 0]
# forbidden = [
#     "",
# ]
choices = [m for m in timm.list_models() if m.find("efficient") >= 0]
choices = []
for model_name in timm.list_models():
    is_of_correct_type = model_name.find("efficient") >= 0
    is_not_tf = not model_name.__contains__("tf")
    if is_of_correct_type and is_not_tf:
        choices.append(model_name)
forbidden = [
    "efficientnet_b1_pruned",
    "efficientnet_b2_pruned",
    "efficientnet_b3_pruned",
]

data = []
for c in choices:
    net_cfg = {"num_classes": 1000, "model_name": c, "pretrained": True}
    print("=" * 50)
    print(c)
    if c not in forbidden:
        model = create_model(**net_cfg)
        total = sum(p.numel() for p in model.parameters())
        print(f"Total params {total:4,d}")
        data.append({"model": c, "size": total})

df = pd.DataFrame(data)
df = df.sort_values(by="size", ascending=True)
df.loc[:, "size"] = df["size"].map("{:,d}".format)
print(df)
