import torch
import torch_pruning as tp
from models.yolo import Detect, IDetect
from models.common import ImplicitA, ImplicitM

model = ...  # Replace with your YOLOv7 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.eval()
print(model)

example_inputs = torch.randn(1, 3, 224, 224).to(device)
imp = tp.importance.MagnitudeImportance(p=2)  # L2 norm pruning

ignored_layers = []
for m in model.modules():
    if isinstance(m, (Detect, IDetect)):
        ignored_layers.append(m.m)

unwrapped_parameters = []

for name, param in model.named_parameters():
    if any(isinstance(model, (ImplicitA, ImplicitM)) for model in model._all_members.values()):
        print(f"Parameter: {name}, Size: {param.size()}")
        unwrapped_parameters.append((param, 1))

iterative_steps = 1  # progressive pruning
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    iterative_steps=iterative_steps,
    pruning_ratio=0.5,  # remove 50% channels
    ignored_layers=ignored_layers,
    unwrapped_parameters=unwrapped_parameters
)

base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
pruner.step()

pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)

print(model)
print("Before Pruning: MACs=%f G, #Params=%f G" % (base_macs / 1e9, base_nparams / 1e9))
print("After Pruning: MACs=%f G, #Params=%f G" % (pruned_macs / 1e9, pruned_nparams / 1e9))
