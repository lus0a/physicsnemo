# LLM-Context.md

## Project Context
- **Name:** NVIDIA PhysicsNeMo
- **Former Name:** NVIDIA Modulus (Do not use "modulus" imports; use "physicsnemo").
- **Framework:** PyTorch-based Physics-ML.
- **Graph Backend:** Currently migrating from DGL to PyTorch Geometric (PyG).

## Key Patterns
- Models are in `physicsnemo.models`.
- We use `input` (or `invar`) and `target` naming conventions for tensors.
