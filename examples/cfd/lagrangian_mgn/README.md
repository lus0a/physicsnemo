# MeshGraphNet with Lagrangian mesh

This is an example of MeshGraphNet for particle-based simulation, based on the
[Learning to Simulate](https://sites.google.com/view/learning-to-simulate/)
work. It demonstrates how to use PhysicsNeMo to train a Graph Neural Network (GNN)
to simulate Lagrangian fluids, solids, and deformable materials.

## Problem overview

In this project, we provide an example of Lagrangian mesh simulation for fluids. The
Lagrangian mesh is particle-based, where vertices represent fluid particles and
edges represent their interactions. Compared to an Eulerian mesh, where the mesh
grid is fixed, a Lagrangian mesh is more flexible since it does not require
tessellating the domain or aligning with boundaries.

As a result, Lagrangian meshes are well-suited for representing complex geometries
and free-boundary problems, such as water splashes and object collisions. However,
a drawback of Lagrangian simulation is that it typically requires smaller time
steps to maintain physically valid prediction.

## Dataset

For this example, we use [DeepMind's particle physics datasets](https://sites.google.com/view/learning-to-simulate).
Some of these datasets contain particle-based simulations of fluid splashing and bouncing
within a box or cube while others use materials like sand or goop.
There are a total of 17 datasets, with some of them listed below:

| Datasets     | Num Particles | Num Time Steps |    dt    | Ground Truth Simulator |
|--------------|---------------|----------------|----------|------------------------|
| Water-3D     | 14k           | 800            | 5ms      | SPH                    |
| Water-2D     | 2k            | 1000           | 2.5ms    | MPM                    |
| WaterRamp    | 2.5k          | 600            | 2.5ms    | MPM                    |
| Sand         | 2k            | 320            | 2.5ms    | MPM                    |
| Goop         | 1.9k          | 400            | 2.5ms    | MPM                    |

See the section **B.1** in the [original paper](https://arxiv.org/abs/2002.09405).

## Model overview and architecture

This model uses MeshGraphNet to capture the dynamics of the fluid system.
The system is represented as a graph, where vertices correspond to fluid particles,
and edges represent their interactions. The model is autoregressive,
utilizing historical data to predict future states. Input features for the vertices
include current position, velocity, node type (e.g., fluid, sand, boundary),
and historical velocity. The model’s output is acceleration, defined as the difference
between current and next velocity. Both velocity and acceleration are derived from
the position sequence and normalized to a standard Gaussian distribution
for consistency.

For computational efficiency, we do not explicitly construct wall nodes for
square or cubic domains. Instead, we assign a wall feature to each interior
particle node, representing its distance from the domain boundaries. For a
system dimensionality of $d = 2$ or $d = 3$, the features are structured
as follows:

- **Node features**:
  - position ($d$)
  - historical velocity ($t \times d$),
    where the number of steps $t$ can be set using `data.num_history` config parameter.
  - one-hot encoding of node type (e.g. 6),
  - wall feature ($2 \times d$)
- **Edge features**: displacement ($d$), distance (1)
- **Node target**: acceleration ($d$)

We construct edges based on a predefined radius, connecting pairs of particle
nodes if their pairwise distance is within this radius. During training, we
shuffle the time sequence and train in batches, with the graph constructed
dynamically within the dataloader. For inference, predictions are rolled out
iteratively, and a new graph is constructed based on previous predictions.
Wall features are computed online during this process. To enhance robustness,
a small amount of noise is added during training.

The model uses a hidden dimensionality of 128 for the encoder, processor, and
decoder. The encoder and decoder each contain two hidden layers, while the
processor consists of ten message-passing layers. We use a batch size of
20 per GPU (for Water dataset), and summation aggregation is applied for
message passing in the processor. The learning rate is set to 0.0001 and decays
using cosine annealing schedule. These hyperparameters can be configured using
command line or in the config file.

## Getting Started

This example uses the lightweight `tfrecord` package to load the data in the `.tfrecord`
format.

Install the requirements using:

```bash
pip install -r requirements.txt
```

To download the data from DeepMind's repo, run:

```bash
cd raw_dataset
bash download_dataset.sh Water /data/
```

This example uses [Hydra](https://hydra.cc/docs/intro/) for [experiment](https://hydra.cc/docs/patterns/configuring_experiments/)
configuration. Hydra offers a convenient way to modify nearly any experiment parameter,
such as dataset settings, model configurations, and optimizer options,
either through the command line or config files.

To view the full set of training script options, run the following command:

```bash
python train.py --help
```

If you encounter issues with the Hydra config, you may receive an error message
that isn’t very helpful. In that case, set the `HYDRA_FULL_ERROR=1` environment
variable for more detailed error information:

```bash
HYDRA_FULL_ERROR=1 python train.py ...
```

To train the model with the Water dataset, run:

```bash
python train.py +experiment=water data.data_dir=/data/Water
```

Progress and loss logs can be monitored using Weights & Biases. To activate that,
set `loggers.wandb.mode` to `online` in the command line:

```bash
python train.py +experiment=water data.data_dir=/data/Water loggers.wandb.mode=online
```

An active Weights & Biases account is required. You will also need to set your
API key either through the command line option `loggers.wandb.wandb_key`
or by using the `WANDB_API_KEY` environment variable:

```bash
export WANDB_API_KEY=key
python train.py ...
```

## Inference

The inference script, `inference.py`, also supports Hydra configuration, ensuring
consistency between training and inference runs.

Once the model is trained, run the following command:

```bash
python inference.py +experiment=water \
    data.data_dir=/data/Water \
    data.test.num_sequences=4 \
    resume_dir=/data/models/lmgn/water \
    output=/data/models/lmgn/water/inference
```

Use the `resume_dir` parameter to specify the location of the model checkpoints.

This will save the predictions for the test dataset as animated `.gif` files in the
`/data/models/lmgn/water/inference/animations` directory.

The script will also generate an `error.png` file,
which displays a visualization of the rollout error.

The results may resemble one of the following, depending on the
material selected for training the model:

![Inference Examples](../../../docs/img/lagrangian_meshgraphnet_multi.png "Inference Examples")

## Codebase Architecture

Here is the architectural layout of the codebase and where you should look to make your modifications:

### Component Breakdown for Developers

#### 1. `datapipe.py` (or `dataset.py`) — *Where the Physics Meets the Graph*

This is the most critical file if you are bringing your own data. Because the DeepMind "Learning to Simulate" dataset is stored in `.tfrecord` format, this file uses the `tensorflow` library to parse the raw byte streams into PyTorch tensors.

* **Graph Construction:** This file is responsible for dynamically building the graph at every timestep. It calculates the Euclidean distance and displacement vectors between particles to create **Edge Features**.
* **Feature Stacking:** It stacks the **Node Features** (current position, historical velocities based on your `num_history` config, and one-hot encoded node types like fluid or boundary).

* **Developer Action:** If you are abandoning `.tfrecord` files for `.csv`, `.h5`, or `.vtp` files, you will completely gut and rewrite the reader logic in this file, ensuring it still returns a PyTorch Geometric (PyG) or DGL graph object.

#### 2. `train.py` — *Training loop*

This is your standard PyTorch entry point. It instantiates the `MeshGraphNet` model, handles distributed data parallel (DDP) scaling, and runs the forward/backward pass.

* **The Loss Function:** By default, the model does *not* predict the next position directly; it predicts the **acceleration** (the difference between the current and next velocity). The loss (usually MSE) is calculated against this normalized acceleration vector.

* **Developer Action:** If you want to add physics-informed constraints (like penalizing mass loss or enforcing incompressibility), you will inject your custom PDE residual loss calculations directly into the training loop inside this file.

#### 3. `conf/` — *Where the Hyperparameters Live*

PhysicsNeMo uses Hydra, meaning you should rarely need to hardcode changes into the Python files.

* **Model Sizing:** You can adjust the Encoder, Processor, and Decoder sizes here. The default sets the hidden dimensionality to 128 and uses 10 message-passing layers in the Processor.
* **Developer Action:** Need to simulate a more complex, long-range physics problem? Increase the `num_processor_layers` in the config so the graph can propagate information further across the mesh in a single timestep.

#### 4. `inference.py` — *Timestepping*

Training a model is a one-step prediction, but simulating physics requires autoregressive rollout (feeding the prediction of $t_1$ back in as the input for $t_2$).

* **Developer Action:** This script handles that iterative looping. If you need to export your predictions into a specific visualization format like `.vtu` for ParaView, you will add your VTK writer functions at the end of the rollout loop here.

## Customization Guide

While this example demonstrates a water-drop simulation using the DeepMind "Learning to Simulate" dataset, the Lagrangian MeshGraphNet architecture is highly versatile. You can extend it to other use cases to model granular flows or  deformable solids. To adapt the Lagrangian MeshGraphNet (MGN) pipeline for your own datasets and applications, use this guide to adapt the codebase to experiment for your custom use case:

### 1. Swapping the Model Backbone: xxxx


### 2. Adapting to Your Own Data

By default, `datapipe.py` uses the `tensorflow` library to parse `.tfrecord` files. If your data is in standard formats like `.h5`, or `.vtp`, you will need to rewrite the data loading logic in `datapipe.py`.

You can import xxx from xxx to create your custom dataloader to dynamically construct a graph at each timestep and output the following core components:

* **Node Features:** A concatenated tensor containing the particle's current position, historical velocities, and a one-hot encoded node type (e.g., fluid, boundary, rigid body).
* **Edge Features:** Computed dynamically based on particle proximity. This typically includes the 3D displacement vector between two connected nodes and their scalar Euclidean distance.
* **Targets:** The ground truth for the next timestep. Note that MGN typically predicts **normalized acceleration** (the difference between the current and next velocity), not direct coordinate positions.

**Tip:** If you are building your own graphs from raw point clouds, xxx function is an efficient way to dynamically generate edge indices based on a physical interaction radius.

### 3. Customizing Input Features

If your simulation requires additional physical parameters (like temperature, mass, or viscosity), you must adjust the feature vectors and model dimensions:

1. **Inject Features:** Append your new physical properties to the node feature array in your data pipeline.
2. **Expand the History Window:** If your physics require a longer temporal context, increase the `data.num_history` parameter in your configuration to look back further in time.
3. **Update Input Dimensions:** Ensure the `model.input_dim` in your Hydra configuration reflects the new total size of your concatenated node feature vector.

### 4. Scaling & Performance Toggles

Lagrangian simulations involve dynamic graphs where connectivity changes continuously. This is computationally expensive, but PhysicsNeMo provides several Hydra configuration levers to optimize performance and VRAM usage.

You can tweak these settings directly in `conf/config.yaml` to experiment:

```yaml
# Architecture Sizing
model:
  hidden_dim: 256              # Default is 128. Increase for complex, multiphase physics.
  num_processor_layers: 15     # Default is 10. Increase to allow information to propagate further across the mesh per timestep.

# Performance Optimizations
  do_concat_trick: True        # Replaces slow tensor concatenations in message passing with an optimized MLP + index + sum operation.
  num_processor_checkpoint_segments: 4  # Enables gradient checkpointing. Increase this if you hit Out-Of-Memory (OOM) errors during training.

# Graph Construction
data:
  radius: 0.015                # The spatial cutoff for creating edges between particles. 
  num_history: 5               # The number of previous timesteps to include in the node features.

```

### 5. Adapting the "Loss function" to New Physics

Data-driven models can occasionally violate fundamental physics. If you want to enforce specific physical laws (like mass conservation or collision penalties), you will need to modify the training loop.

* Navigate to `train.py`.
* Locate the forward pass and MSE loss calculation (which compares predicted acceleration against ground-truth acceleration).
* Inject your custom PDE residual or penalty terms here. Calculate the physical constraint violation based on the model's predicted next-step positions, multiply it by a weighting scalar ($\lambda$), and add it to the total training loss.


## References

- [Learning to simulate complex physicswith graph networks](https://arxiv.org/abs/2002.09405)
- [Dataset](https://sites.google.com/view/learning-to-simulate)
- [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)
