from timm.models import create_model
from pactl.nn.projectors import find_all_batch_norm
from pactl.nn.projectors import find_locations_from_condition

model_name = "mobilevit_xxs"
net_cfg = {"num_classes": 1000, "model_name": model_name, "pretrained": True}
model = create_model(**net_cfg)
count = find_all_batch_norm(model)
print(f"Net {model_name} has: {count:,d} Batch Norm params")

aux = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
names, params = zip(*aux)
names, params = list(names), list(params)

ids = find_locations_from_condition(names, params, lambda x: x.find("bn") >= 0)
print(f"Batch Norm params found {len(ids):,d}")
