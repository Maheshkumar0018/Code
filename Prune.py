import torch
import torch_pruning as tp
from models.yolo import Detect, IDetect
from models.common import ImplicitA, ImplicitM

class Pruner:
    def __init__(self, model, device, sparsity, prune_norm='L2'):
        model.eval()
        example_inputs = torch.randn(1, 3, 416, 416).to(device)  # Adjust input dimensions based on your YOLOv7 model
        imp = tp.importance.MagnitudeImportance(p=2 if prune_norm == 'L2' else 1)  # L2 norm pruning

        ignored_layers = []
        for m in model.modules():
            if isinstance(m, (Detect, IDetect)):
                ignored_layers.append(m.m)
        unwrapped_parameters = []
        for m in model.modules():
            if isinstance(m, (ImplicitA, ImplicitM)):
                unwrapped_parameters.append((m.implicit, 1))  # pruning 1st dimension of implicit matrix

        self.pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            ch_sparsity=sparsity,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters
        )
        self.sparsity = sparsity

    def prune(self):
        example_inputs = torch.randn(1, 3, 416, 416).to(device)  # Adjust input dimensions based on your YOLOv7 model

        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

        self.pruner.step()

        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print("Pruning Sparsity=%f" % self.sparsity)
        print("Before Pruning: MACs=%f, #Params=%f" % (base_macs, base_nparams))
        print("After Pruning: MACs=%f, #Params=%f" % (pruned_macs, pruned_nparams))

# Assuming you have a custom YOLOv7 model saved with weights
custom_yolov7_model = ...  # Load your custom YOLOv7 model here

# Create an instance of the pruner class
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pruner_instance = Pruner(custom_yolov7_model, device=device, sparsity=0.5)

# Run pruning
pruner_instance.prune()
