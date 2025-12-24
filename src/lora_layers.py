"""
LoRA (Low-Rank Adaptation) implementation for PINN fine-tuning

Implements parameter-efficient transfer learning by adding low-rank
decomposition matrices to existing linear layers.
"""

import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps an existing Linear layer

    Adds trainable low-rank matrices A and B such that:
    h = Wx + (alpha/r) * BAx

    where r is the rank, and only A and B are trained during fine-tuning.
    """

    def __init__(self, linear_layer, r=4, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Store original layer (frozen)
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # Get device from linear layer
        device = linear_layer.weight.device

        # LoRA matrices (on same device as linear layer)
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=device))

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialize A with kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original layer output (frozen)
        result = self.linear(x)

        # LoRA adaptation
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        """Merge LoRA weights into original linear layer for inference"""
        if self.r > 0:
            delta_w = self.lora_B @ self.lora_A * self.scaling
            self.linear.weight.data += delta_w
            # Zero out LoRA matrices after merging
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()


def add_lora_to_linear(module, target_names, r=4, lora_alpha=16, lora_dropout=0.0):
    """
    Add LoRA layers to specified linear layers in a module

    Args:
        module: PyTorch module to modify
        target_names: List of layer names to add LoRA to (e.g., ['layers.0', 'layers.2'])
        r: LoRA rank
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout rate for LoRA

    Returns:
        Modified module with LoRA layers
        Number of trainable LoRA parameters
    """
    lora_params = 0

    # Handle Sequential modules specially
    if isinstance(module, nn.Sequential):
        new_layers = []
        for i, child in enumerate(module):
            if isinstance(child, nn.Linear):
                # Replace with LoRA layer
                lora_layer = LoRALayer(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                new_layers.append(lora_layer)

                # Count parameters
                lora_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
                print(f"  Added LoRA to layer {i}: {child.in_features}→{child.out_features}, "
                      f"+{lora_layer.lora_A.numel() + lora_layer.lora_B.numel()} params")
            else:
                new_layers.append(child)

        # Replace Sequential
        module.__init__(*new_layers)
    else:
        # Regular module traversal
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(target in name for target in target_names):
                # Replace with LoRA layer
                lora_layer = LoRALayer(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                setattr(module, name, lora_layer)

                # Count parameters
                lora_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
                print(f"  Added LoRA to {name}: {child.in_features}→{child.out_features}, "
                      f"+{lora_layer.lora_A.numel() + lora_layer.lora_B.numel()} params")
            else:
                # Recursively apply to children
                child_params = add_lora_to_linear(child, target_names, r, lora_alpha, lora_dropout)
                lora_params += child_params

    return lora_params


def add_lora_to_model(model, r=4, lora_alpha=16, lora_dropout=0.0, target_modules=None):
    """
    Add LoRA to PINN model's solution_u encoder

    Args:
        model: PINN model
        r: LoRA rank
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout rate
        target_modules: Not used (kept for compatibility)

    Returns:
        Number of trainable LoRA parameters
    """
    print(f"\nAdding LoRA (r={r}, alpha={lora_alpha}) to solution_u.encoder:")

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Add LoRA to encoder (Sequential)
    # Handle both cases: encoder.net (original) and encoder (massive model)
    encoder = getattr(model.solution_u.encoder, 'net', model.solution_u.encoder)
    lora_params = add_lora_to_linear(
        encoder,
        [],  # Not used for Sequential
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )

    print(f"Total LoRA parameters: {lora_params}")

    return lora_params


def get_lora_parameters(model):
    """Get all LoRA parameters from a model"""
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_trainable_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_trainable_parameters(model):
    """Print trainable parameters breakdown"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    return trainable_params, total_params
