for name, module in model.named_modules():
    if isinstance(module, (ImplicitA, ImplicitM)):
        print(f"Module: {name}, Type: {type(module)}")
        for param_name, param in module.named_parameters():
            print(f"  Parameter: {param_name}, Size: {param.size()}")
            unwrapped_parameters.append((param, 1))
