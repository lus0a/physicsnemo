# LLM-Context.md

## Project Context
- **Name:** NVIDIA PhysicsNeMo
- **Former Name:** NVIDIA Modulus (Do not use "modulus" imports; use "physicsnemo").
- **Framework:** PyTorch-based Physics-ML.
- **Graph Backend:** Currently migrating from DGL to PyTorch Geometric (PyG).

## Key Patterns
- Models are in `physicsnemo.models`.
- We use `input` (or `invar`) and `target` naming conventions for tensors.

---

## Core Design Patterns

### 1. Registry Pattern with Borg Singleton

**Purpose:** Centralized model discovery and instantiation

PhysicsNeMo uses a **centralized model registry** that enables models to be discovered and instantiated by name:

**Implementation:**
- **Borg Singleton Pattern**: All `ModelRegistry` instances share the same state (not traditional singleton)
- **Entry Points System**: Models can be registered via `pyproject.toml` entry points or programmatically
- **Lazy Loading**: Models are loaded on-demand when first accessed
- **Factory Method**: `registry.factory(name)` creates model instances by name

**Example:**

```python
from physicsnemo.core.registry import ModelRegistry

# All registry instances share the same state
registry = ModelRegistry()

# Lazy loading - model loaded only when accessed
ModelClass = registry.factory('AFNO')
model = ModelClass(input_dim=64, output_dim=32)

# Register custom models
class MyCustomModel(Module):
    pass

registry.register(MyCustomModel, 'MyModel')
```

**Benefits:**
- Discoverability: Users can list all available models
- Plugin architecture: Third-party packages can register models
- Backward compatibility: Handles "modulus" to "physicsnemo" namespace migration

**File Location:** `physicsnemo/core/registry.py`

---

### 2. Module Hierarchy with Enhanced Capabilities

**Purpose:** Extend PyTorch Module with framework-specific features

All models inherit from `physicsnemo.Module` (not `torch.nn.Module` directly) which adds essential capabilities:

**Key Features:**

1. **Custom Serialization**
   - `.mdlus` checkpoint format (zip-based, replacing legacy tar)
   - Supports nested `physicsnemo.Module` instances
   - JSON-serializable `__init__` arguments captured automatically

2. **Versioning System**
   - `__model_checkpoint_version__`: Track API versions
   - Backward compatibility mappers for loading old checkpoints
   - Clear warning messages when loading older versions

3. **Metadata Tracking**
   - `ModelMetaData` for optimization flags (JIT, AMP, CUDA graphs)
   - Deployment target information (ONNX, TensorRT)
   - Physics-informed features (auto-grad, functional)

4. **Automatic Instantiation Tracking**
   - `__new__` captures all `__init__` arguments
   - Enables checkpoint loading without manual configuration
   - Supports nested module serialization

5. **Device Management**
   - Built-in `device` property via device_buffer
   - Automatic device detection for checkpoint loading

**Example:**

```python
from physicsnemo.core import Module, ModelMetaData

class MyModel(Module):
    __model_checkpoint_version__ = "1.0.0"
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(meta=ModelMetaData())
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# Save checkpoint
model.save("checkpoint.mdlus")

# Load checkpoint - automatically instantiates with correct args
loaded_model = Module.from_checkpoint("checkpoint.mdlus")
```

**Pattern Type:** Wrapper/Decorator pattern that extends PyTorch's Module

**File Location:** `physicsnemo/core/module.py`

---
### 3. Configuration-Driven Architecture

**Purpose:** Flexible, composable model configuration

PhysicsNeMo integrates **Hydra-core** for sophisticated configuration management:

**Key Principles:**

1. **Hierarchical Configs**: Compose complex configurations from simple parts
2. **Type-Safe**: OmegaConf provides runtime type checking
3. **Explicit over Implicit**: Prefer explicit Dict parameters over `**kwargs`
4. **Dependency Injection**: Pass instantiated modules, not string type names

**Anti-Pattern (String-Based Selection):**

```python
# WRONG: String-based class selection with many options
class MyModel(Module):
    def __init__(self, encoder_type: str = "transformer"):
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder()
        elif encoder_type == "cnn":
            self.encoder = CNNEncoder()
        # ... 10+ more options (hard to type-check, debug, test)
```

**Good Pattern (Dependency Injection):**

```python
# GOOD: Instance injection for flexibility
class MyModel(Module):
    def __init__(
        self,
        encoder: Module,  # Pass instance, not string
        decoder: Module
    ):
        self.encoder = encoder
        self.decoder = decoder

# Usage with Hydra config
model = MyModel(
    encoder=TransformerEncoder(dim=128, layers=6),
    decoder=TransformerDecoder(dim=128, layers=6)
)
```

**Configuration Passing Pattern:**

```python
# GOOD: Explicit Dict for submodule config
class MyModel(Module):
    def __init__(
        self,
        input_dim: int,
        encoder_config: Optional[Dict[str, Any]] = None
    ):
        encoder_config = encoder_config or {}
        self.encoder = Encoder(input_dim=input_dim, **encoder_config)

# Usage
model = MyModel(
    input_dim=64,
    encoder_config={"hidden_dim": 128, "num_layers": 3}
)
```

**Pattern Type:** Dependency Injection + Strategy Pattern

**Related Rules:** MOD-009, MOD-010

---

### 4. Separation of Concerns - Layer vs Model Organization

**Purpose:** Clear boundaries between reusable components and complete models

**Architectural Boundaries:**

```
physicsnemo/
├── nn/                    # Reusable building blocks
│   ├── attention_layers.py
│   ├── conv_layers.py
│   ├── fourier_layers.py
│   └── gnn_layers/
│
├── models/               # Complete domain-specific models  
│   ├── afno/
│   ├── fno/
│   ├── graphcast/
│   └── meshgraphnet/
│
└── experimental/         # Models under development
    ├── nn/              # Experimental layers
    └── models/          # Experimental models
```

**Organization Rules:**

1. **Reusable Layers** → `physicsnemo/nn/`
   - Building blocks used across multiple models
   - Example: `MultiHeadAttention`, `FourierLayer`, `FullyConnected`
   - Must be imported in `physicsnemo/nn/__init__.py`

2. **Complete Models** → `physicsnemo/models/`
   - Domain-specific or modality-specific models
   - Composed of multiple layers
   - Must be imported in `physicsnemo/models/__init__.py`

3. **Self-Contained Modules**
   - Model-specific utilities live with the model
   - Either single file or subdirectory structure
   - Avoid flat organization with scattered utility files

**Example Structure:**

```python
# Good: Single self-contained file
# File: physicsnemo/models/my_simple_model.py

def _compute_attention_mask(seq_length: int) -> torch.Tensor:
    """Helper function specific to MySimpleModel."""
    ...

class MySimpleModel(Module):
    """Model with utilities in same file."""
    pass

# Good: Subdirectory for complex models
# physicsnemo/models/graphcast/
#   ├── graph_cast_net.py      # Main model
#   ├── graph_cast_processor.py
#   └── utils/
#       ├── graph_utils.py
#       └── icosahedral_mesh.py
```

**Pattern Type:** Component-Based Architecture with clear ownership

**Related Rules:** MOD-000a, MOD-000b, MOD-004

---

## Architectural Patterns

### 5. Data Pipeline Pattern

**Purpose:** Modular, domain-specific data loading

PhysicsNeMo provides a rich set of **domain-specific datapipes**:

**Organization:**

```
physicsnemo/datapipes/
├── climate/              # ERA5, HEALPix, synthetic climate data
├── cfd/                 # CFD simulation datasets  
├── gnn/                 # Graph-based datasets (Ahmed body, vortex shedding)
├── cae/                 # CAE/FEA datasets
├── healpix/             # HEALPix grid datasets
└── benchmarks/          # Standardized benchmark datasets
```

**Key Features:**

1. **Domain Specialization**: Each domain has custom datapipe implementations
2. **Composable Operations**: Chain transformations on data
3. **Multiple Storage Backends**: Local, S3, HTTP via `fsspec`
4. **Format Support**: HDF5, NetCDF, Zarr, VTK, and more

**Example:**

```python
from physicsnemo.datapipes.climate import ERA5HDF5Datapipe

# Domain-specific datapipe with built-in transformations
datapipe = ERA5HDF5Datapipe(
    data_dir="/path/to/era5",
    variables=["temperature", "pressure"],
    time_range=("2020-01-01", "2020-12-31"),
    normalize=True
)

# Iterate over batches
for batch in datapipe:
    # batch contains normalized climate data
    pass
```

**Pattern Type:** Strategy Pattern for domain-specific data handling

**File Location:** `physicsnemo/datapipes/`

---

### 6. Distributed Computing Patterns

**Purpose:** Scale models across multiple GPUs/nodes

Multiple parallelism strategies with unified interface:

**Parallelism Types:**

1. **Data Parallelism** (Standard PyTorch DDP)
   - Replicate model across devices
   - Gradient synchronization

2. **Domain Parallelism** (Custom Implementation)
   - `ShardTensor`: Spatial decomposition of data
   - Halo exchange for ghost cells
   - Ring communication patterns
   - Custom distributed operations

3. **Model Parallelism** (via PyTorch)
   - Partition model across devices
   - Pipeline parallelism support

**Key Components:**

```
physicsnemo/distributed/
├── manager.py           # Process group management
├── autograd.py         # Distributed automatic differentiation
├── fft.py              # Distributed FFT operations
└── mappings.py         # Tensor distribution strategies

physicsnemo/domain_parallel/
├── shard_tensor.py           # ShardTensor implementation
├── shard_utils/
│   ├── halo.py              # Halo exchange
│   ├── ring.py              # Ring communication
│   ├── attention_patches.py # Distributed attention
│   └── conv_patches.py      # Distributed convolution
```

**Example:**

```python
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor

# Initialize distributed environment
manager = DistributedManager()

# Create sharded tensor for domain decomposition
x_sharded = ShardTensor.from_tensor(
    x, 
    shard_dims=[2, 3],  # Shard spatial dimensions
    process_group=manager.spatial_group
)

# Operations automatically handle communication
output = model(x_sharded)  # Halo exchange happens automatically
```

**Pattern Type:** Facade Pattern hiding distributed complexity

**File Location:** `physicsnemo/distributed/`, `physicsnemo/domain_parallel/`

---

### 7. Dependency Management Pattern

**Purpose:** Centralized, graceful handling of optional dependencies

**Key Mechanisms:**

1. **`check_min_version(package, version, hard_fail=False)`**
   - Check availability without importing
   - Returns boolean for soft requirements
   - Raises error for hard requirements

2. **`@require_version(package, version)`**
   - Decorator to protect version-specific features
   - Provides clear error message if version insufficient

3. **`pyproject.toml` as Single Source of Truth**
   - All dependencies declared centrally
   - Optional dependencies in separate groups

**Example:**

```python
from physicsnemo.core.version_check import check_min_version, require_version

# Check optional dependency without importing
APEX_AVAILABLE = check_min_version("apex", "0.1.0", hard_fail=False)

class MyModel(Module):
    def __init__(self, use_apex: bool = False):
        super().__init__()
        self.use_apex = use_apex
        
        if use_apex and not APEX_AVAILABLE:
            raise RuntimeError(
                "apex is required for use_apex=True but is not installed. "
                "Install with: pip install apex>=0.1.0"
            )
        
        if use_apex:
            import apex  # Import only when needed
            self.fused_layer = apex.FusedLayer()

# Protect version-specific features
class AdvancedModel(Module):
    @require_version("torch", "2.4.0")
    def use_device_mesh(self):
        """This feature requires torch>=2.4.0."""
        from torch.distributed.device_mesh import DeviceMesh
        ...
```

**Benefits:**
- Graceful degradation when optional deps missing
- Clear, actionable error messages
- Prevents import errors at module load time
- Centralized dependency specification

**Pattern Type:** Lazy Initialization + Dependency Injection

**Related Rule:** MOD-011

---

### 8. Validation Guard Pattern

**Purpose:** Early error detection with compilation compatibility

Shape validation at API boundaries with **torch.compile** awareness:

**Pattern:**

```python
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Forward pass with shape validation."""
    
    ### Input validation
    # Skip validation when running under torch.compile
    if not torch.compiler.is_compiling():
        # Extract expected dimensions
        B, C, H, W = x.shape if x.ndim == 4 else (None, None, None, None)
        
        # Validate x shape
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), "
                f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
            )
        
        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {C}"
            )
        
        # Validate optional mask
        if mask is not None:
            if mask.shape != (B, H, W):
                raise ValueError(
                    f"Expected mask shape ({B}, {H}, {W}), "
                    f"got {tuple(mask.shape)}"
                )
    
    # Actual computation after validation
    return self._process(x, mask)
```

**Key Principles:**

1. **Fail-Fast**: Validate before any computation
2. **Clear Messages**: Include expected vs actual in error messages
3. **Compilation Compatible**: Guard with `torch.compiler.is_compiling()`
4. **Container Validation**: Check length, keys, and tensor shapes

**Benefits:**
- Debug-time safety with clear error messages
- Production performance (validation skipped in compiled code)
- Catches shape mismatches at API boundary

**Pattern Type:** Guard Clause pattern with compiler awareness

**Related Rule:** MOD-005

---

### 9. Type-Safe Tensor Annotations

**Purpose:** Machine-readable shape documentation with optional runtime validation

PhysicsNeMo uses **JAXTyping** for explicit tensor shape annotations:

**Pattern:**

```python
from jaxtyping import Float
import torch

class MyConvNet(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3)
    
    def forward(
        self,
        x: Float[torch.Tensor, "batch in_channels height width"]
    ) -> Float[torch.Tensor, "batch out_channels height width"]:
        """Process input with convolution."""
        return self.conv(x)
```

**Benefits:**

1. **Self-Documenting**: Shape information in type hints
2. **IDE Support**: Better autocompletion and error detection
3. **Runtime Validation**: Optional runtime checking when enabled
4. **Static Analysis**: Tools can verify shape consistency

**Naming Conventions:**
- Use descriptive names: `"batch channels height width"`
- Or short forms: `"b c h w"`
- Consistency with docstring math notation

**Pattern Type:** Type-Level Documentation with optional runtime validation

**Related Rule:** MOD-006

---

### 10. Metrics and Evaluation Pattern

**Purpose:** Domain-specific, composable evaluation metrics

Modular metrics organized by domain:

**Organization:**

```
physicsnemo/metrics/
├── climate/              # ACC, EFI, HEALPix losses
├── diffusion/            # FID, diffusion-specific losses
├── general/              # MSE, CRPS, Wasserstein, ensemble metrics
└── cae/                  # CFD metrics, integral quantities
```

**Key Features:**

1. **Domain Specialization**: Metrics encode domain knowledge
2. **Composable Reductions**: Flexible aggregation strategies
3. **Physics-Informed**: Incorporate physical constraints
4. **Distributed-Aware**: Support for multi-GPU reduction

**Example:**

```python
from physicsnemo.metrics.climate import ACC
from physicsnemo.metrics.general import CRPSLoss

# Anomaly Correlation Coefficient for climate
acc_metric = ACC(mean_dims=[2, 3], reduce_dims=[0])

# Continuous Ranked Probability Score
crps = CRPSLoss(reduction='mean')

# Use in training loop
for batch in dataloader:
    pred = model(batch['input'])
    acc_score = acc_metric(pred, batch['target'], batch['climatology'])
    crps_loss = crps(pred, batch['target'])
```

**Pattern Type:** Strategy Pattern for domain-specific evaluation

**File Location:** `physicsnemo/metrics/`

---

### 11. Documentation-as-Code Pattern

**Purpose:** Living documentation that's tested and maintained

Comprehensive docstring standards enforced by CI:

**Required Sections (Classes):**

1. **Parameters**: All `__init__` arguments with types and defaults
2. **Forward**: Input tensors for forward method
3. **Outputs**: Return values with shapes
4. **Examples**: Executable code (CI-tested)

**Required Sections (Methods):**

1. **Parameters**: All arguments with types
2. **Returns**: Return values with types

**Format Requirements:**

```python
class MyEncoder(Module):
    r"""
    A simple encoder network.
    
    This implementation uses multi-head attention. See
    :class:`~physicsnemo.nn.MultiHeadAttention` for details.
    
    Parameters
    ----
    input_dim : int
        Dimension of input features.
    output_dim : int
        Dimension of output features.
    hidden_dim : int, optional, default=128
        Dimension of hidden layer.
    
    Forward
    ----
    x : torch.Tensor
        Input tensor of shape :math:`(B, D_{in})` where :math:`B` is
        batch size and :math:`D_{in}` is input dimension.
    
    Outputs
    ----
    torch.Tensor
        Output tensor of shape :math:`(B, D_{out})`.
    
    Examples
    -----
    >>> import torch
    >>> from physicsnemo.models import MyEncoder
    >>> 
    >>> # Create model
    >>> model = MyEncoder(input_dim=784, output_dim=128)
    >>> 
    >>> # Process a batch
    >>> x = torch.randn(32, 784)
    >>> output = model(x)
    >>> output.shape
    torch.Size([32, 128])
    """
    pass
```

**Key Standards:**

1. **Raw Strings**: Use `r"""` for LaTeX compatibility
2. **LaTeX Math**: `:math:`(B, D)`` for tensor shapes
3. **Double Backticks**: Use ``code`` for inline code
4. **Cross-References**: Link to other classes/functions
5. **CI-Tested Examples**: Examples automatically tested

**Pattern Type:** Living Documentation

**Related Rules:** MOD-003a through MOD-003k

---

### 12. Testing Pattern - Triple Verification

**Purpose:** Comprehensive testing before production promotion

Every production model requires **three types of tests**:

**1. Constructor Tests**

```python
@pytest.mark.parametrize("config", ["default", "custom"])
def test_my_model_constructor(config):
    """Test model instantiation and attributes."""
    if config == "default":
        model = MyModel(input_dim=64, output_dim=32)
        assert model.hidden_dim == 128  # Check default
    else:
        model = MyModel(input_dim=64, output_dim=32, hidden_dim=256)
        assert model.hidden_dim == 256  # Check custom
```

**2. Non-Regression Tests**

```python
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_my_model_non_regression(device):
    """Test against reference outputs."""
    model = _instantiate_model(MyModel, input_dim=64, output_dim=32)
    model = model.to(device)
    
    # Load reference data
    data = torch.load(f"test/models/data/my_model_v1.0.pth")
    x = data["x"].to(device)
    out_ref = data["out"].to(device)
    
    # Compare actual values (not just shapes!)
    out = model(x)
    assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5)
```

**3. Checkpoint Tests**

```python
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_my_model_from_checkpoint(device):
    """Test checkpoint loading."""
    model = Module.from_checkpoint(
        "test/models/data/my_model_v1.0.mdlus"
    ).to(device)
    
    # Verify attributes
    assert model.input_dim == 64
    
    # Verify outputs match reference
    data = torch.load("test/models/data/my_model_v1.0.pth")
    x = data["x"].to(device)
    out_ref = data["out"].to(device)
    out = model(x)
    assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5)
```

**Critical Rules:**

- Models cannot move from experimental → production without all three tests
- Test data must have realistic shapes (no singletons except batch)
- Use pytest parameterization for multiple configurations
- Compare actual tensor values, not just shapes

**Pattern Type:** Test Pyramid with regression protection

**Related Rules:** MOD-008a, MOD-008b, MOD-008c

---

### 13. Metadata Pattern

**Purpose:** Track model capabilities and optimization flags

`ModelMetaData` dataclass provides runtime information:

**Structure:**

```python
@dataclass
class ModelMetaData:
    # Optimization
    jit: bool = False            # TorchScript JIT compilation
    cuda_graphs: bool = False    # CUDA graph capture
    amp: bool = False            # Automatic Mixed Precision
    amp_cpu: bool = None         # AMP for CPU
    amp_gpu: bool = None         # AMP for GPU
    torch_fx: bool = False       # Torch FX support
    
    # Data type
    bf16: bool = False           # BFloat16 support
    
    # Inference
    onnx: bool = False           # ONNX export support
    onnx_gpu: bool = None        # ONNX GPU deployment
    onnx_cpu: bool = None        # ONNX CPU deployment
    onnx_runtime: bool = False   # ONNX Runtime compatibility
    trt: bool = False            # TensorRT support
    
    # Physics informed
    var_dim: int = -1            # Variable dimension
    func_torch: bool = False     # Functional torch API
    auto_grad: bool = False      # Supports automatic differentiation
```

**Usage:**

```python
class MyModel(Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(meta=ModelMetaData(
            jit=True,            # Can be JIT compiled
            amp=True,            # Supports AMP
            onnx=True,           # Can export to ONNX
            auto_grad=True       # Differentiable
        ))
        ...
```

**Benefits:**
- Runtime capability checking
- Optimization hint for training loops
- Deployment target validation
- Documentation of model features

**Pattern Type:** Capability Metadata pattern

**File Location:** `physicsnemo/core/meta.py`

---

### 14. Entry Points and Extensibility

**Purpose:** Plugin architecture for third-party extensions

PhysicsNeMo supports **plugin-based model registration**:

**Entry Points in `pyproject.toml`:**

```toml
[project.entry-points."physicsnemo.models"]
MyCustomModel = "my_package.models:MyCustomModel"
AFNO = "physicsnemo.models.afno:AFNO"
FNO = "physicsnemo.models.fno:FNO"
```

**Programmatic Registration:**

```python
# Option 1: Use register=True in class definition
class MyModel(Module, register=True):
    def __init__(self, hidden_dim: int = 64):
        super().__init__(meta=ModelMetaData())
        self.hidden_dim = hidden_dim

# Option 2: Manual registration
from physicsnemo.core.registry import ModelRegistry

registry = ModelRegistry()
registry.register(MyModel, 'MyModel')

# Both approaches enable:
ModelClass = registry.factory('MyModel')
model = ModelClass(hidden_dim=128)
```

**Benefits:**
- Third-party packages can extend PhysicsNeMo
- Discoverability of available models
- Namespace management (physicsnemo.models vs external)
- Lazy loading for faster imports

**Pattern Type:** Plugin Architecture

**File Location:** `physicsnemo/core/registry.py`

---

## Code Quality Patterns

### 15. High-Level Comments Pattern

**Purpose:** Semantic explanations for complex tensor operations

**Principles:**

1. **Semantic over Syntactic**: Explain *what* not *how*
2. **Shape Annotations**: Inline tensor shapes in comments
3. **Consistency**: Match docstring notation

**Example:**

```python
def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Process input with context conditioning."""
    
    # Encode input features
    h = self.encoder(x)  # (B, C_enc, H, W)
    
    # Combine with context information
    c = self.context_proj(context)  # (B, C_enc)
    c = c[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])  # (B, C_enc, H, W)
    h = torch.cat([h, c], dim=1)  # (B, 2*C_enc, H, W)
    
    # Apply attention mechanism
    h = self.attention(h)  # (B, 2*C_enc, H, W)
    
    # Decode to output
    out = self.decoder(h)  # (B, C_out, H, W)
    
    return out
```

**Anti-Pattern (Too Low-Level):**

```python
# WRONG: Syntactic, low-level comments
def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    # Pass x through encoder layer
    h = self.encoder(x)
    # Project context using linear layer
    c = self.context_proj(context)
    # Add two None dimensions and expand
    c = c[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])
```

**Related Rule:** MOD-003k

---

### 17. Comprehensive Error Messages

**Purpose:** Debugging-friendly error messages

**Pattern:**

```python
# Include expected vs actual
if x.ndim != 4:
    raise ValueError(
        f"Expected 4D input tensor (B, C, H, W), "
        f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
    )

# Include parameter values
if C != self.in_channels:
    raise ValueError(
        f"Expected {self.in_channels} input channels, got {C} channels"
    )

# Include actionable suggestions
if use_apex and not APEX_AVAILABLE:
    raise RuntimeError(
        "apex is required for use_apex=True but is not installed. "
        "Install with: pip install apex>=0.1.0"
    )
```

**Benefits:**
- Users can immediately understand what went wrong
- Clear path to resolution
- Reduces support burden

---

## Design Pattern Summary Table

| Pattern | Purpose | Key Benefit | Location |
|---------|---------|-------------|----------|
| Registry (Borg Singleton) | Model discovery | Plugin architecture | `core/registry.py` |
| Enhanced Module | PyTorch extension | Serialization + versioning | `core/module.py` |
| Versioning | Backward compat | Safe evolution | `core/module.py` |
| Config-Driven | Flexible composition | Type-safe configs | Throughout |
| Layer/Model Separation | Clear boundaries | Reusability | `nn/`, `models/` |
| Data Pipeline | Domain-specific loading | Modularity | `datapipes/` |
| Distributed Computing | Multi-GPU scaling | Transparent parallelism | `distributed/` |
| Dependency Management | Optional deps | Graceful degradation | `core/version_check.py` |
| Validation Guards | Early error detection | Clear error messages | Throughout |
| JAXTyping | Shape documentation | Type safety | Throughout |
| Metrics | Domain-specific eval | Physics-informed | `metrics/` |
| Doc-as-Code | Living documentation | CI-tested examples | Throughout |
| Triple Testing | Quality assurance | Regression protection | `test/` |
| Metadata | Capability tracking | Runtime adaptation | `core/meta.py` |
| Entry Points | Extensibility | Plugin system | `core/registry.py` |

---

## Conclusion

PhysicsNeMo's design patterns create a **scientific computing framework** that balances:

- **Flexibility** (for researchers) with **Stability** (for production)
- **Performance** (compilation-friendly) with **Debuggability** (clear errors)
- **Extensibility** (plugins) with **Consistency** (strict coding standards)

These patterns reflect lessons learned from deploying ML models in production scientific computing environments where **reproducibility**, **backward compatibility**, and **clear error messages** are not optional—they're requirements.

---

**Last Updated:** December 19, 2025

