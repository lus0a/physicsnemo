# `physicsnemo.experimental.integrations.newton`

PhysicsNeMo adapters for [Newton](https://github.com/newton-physics/newton), a
GPU-accelerated physics engine built on Warp.

This API is experimental and may change as the integration matures.

Newton owns the physical world and numerical time stepping.

PhysicsNeMo owns the models and reusable physics-AI workflows.
This optional integration connects them through live Warp-to-Torch views
without an unnecessary device-to-host copy.

Runnable examples live in [examples/newton](../../../../examples/newton). The
experimental API guide is the
[Newton integration documentation](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/physicsnemo.experimental.integrations.newton.html).

## Mental model

A Newton simulation has a few core objects:

| Newton object | Meaning |
| --- | --- |
| `model` | Mostly fixed bodies, particles, shapes, masses, joints, and materials |
| `state` | Positions, orientations, and velocities at one instant |
| `control` | Actuator targets, forces, or other inputs applied while advancing the world |
| collision pipeline / `contacts` | Interactions found for the current state |
| `solver` | Numerical method that advances `state_in` to `state_out` |

Newton can replicate one scene into many parallel **worlds**: independent
copies of the same physics, batched on one device and advanced by a single
solver call. Shapes such as `[worlds, features]` below refer to that batch.

PhysicsNeMo is built on PyTorch, so users retain ordinary tensors, modules,
losses, autograd, and optimizers. This integration adds the reusable parts a raw
PyTorch script does not know:

| Need | Public integration support |
| --- | --- |
| Drive a Newton scene headlessly | `NewtonEnv`, `load_example_scene` |
| Read/write live Newton arrays as Torch tensors | `field_to_torch`, `particles`, `bodies`, `joints` |
| Differentiate through a Newton rollout | `differentiable_rollout` |
| Train one-step and free-running dynamics from trajectories | `BPTTSurrogate`, `trajectory_dataset` |
| Use Newton as an optimization oracle | `DesignSurrogate`, `optimize_design` |
| Describe reusable physical design variables | `physicsnemo.experimental.integrations.newton.DesignSpace` |
| Share rigid geometry between Newton and neural features | `NewtonRigidObject`, `NewtonPrimitive`, `NewtonMesh` |
| Optimize one shared design over grouped candidates | `optimize_grouped_design` |
| Train candidate/action ranking within each group | `grouped_candidate_ranking_loss` |
| Send only the best learned candidates to verification | `shortlist_grouped_candidates` |
| Harvest and thin candidates along optimization paths | `GroupedDesignResult.trajectory_candidates`, `physicsnemo.experimental.integrations.newton.select_diverse_designs` |
| Accept a learned proposal only after authoritative verification | `physicsnemo.experimental.integrations.newton.select_verified_design` |
| Learn and deploy a Newton step | `NeRDProblem`, `NeRDControlInput`, `fit_nerd`, `TrainedNeRDModel` |
| Scale a NeRD run | automatic trajectory sharding, global statistics, and DDP |

Newton remains the reference solver during *teacher* generation (the
ground-truth solver runs that produce training data) and validation. A learned
model replaces a solver step only when explicitly deployed through
`NewtonStepModel`.

## First rollout

Install the optional integration first (see
[the installation note](#installation) at the end of this document; in short,
`pip install "nvidia-physicsnemo[cu13,newton]"`, or `cu12` for CUDA 12).

```python
import torch
from newton.examples.diffsim.example_diffsim_ball import Example as BallScene

from physicsnemo.experimental.integrations.newton import (
    NewtonEnv,
    particles,
)


def observe(state):
    ball = particles(state)
    return torch.cat((ball.positions[0], ball.velocities[0]))


env = NewtonEnv.from_example(
    BallScene,
    observe=observe,
    substeps=4,
    collide_on_reset=True,
    # This stock scene authors its ground-plane contacts once at reset and
    # reuses them, so skip per-substep recollision.
    collide_each_substep=False,
)
env.reset()
trajectory = env.rollout(steps=60).observations
```

`NewtonEnv` owns reset, force clearing, collisions, substeps, solver calls, and
state-buffer ping-pong. `from_example(...)` wraps the example through
`from_scene(...)`, which resets from the scene's authored initial state and
refreshes contacts before each substep when the scene exposes a collision
pipeline. A one-shot contact is computed once (here, against the fixed ground
plane at reset) and reused for the whole rollout instead of being recomputed
every substep. Because this stock ball scene authors such contacts, pass
`collide_each_substep=False` to skip per-substep recollision; for scenes whose
contacts change during a step, leave it at its default (true when the scene has
a collision pipeline). The observation remains plain application code.

Inputs that must be reapplied after Newton clears forces can use
`before_substep`:

```python
def apply_force(state, control, contacts, dt, substep):
    control.joint_f.fill_(10.0)


env.step(before_substep=apply_force)
```

## State on the learning boundary

Newton packs different physics into different fields:

| Physics | Newton fields | Readable live view |
| --- | --- | --- |
| Particles, soft bodies, MPM | `particle_q`, `particle_qd` | `particles(state)` |
| Rigid bodies and rigid-body cables | `body_q`, `body_qd` | `bodies(state)` |
| Articulations | `joint_q`, `joint_qd` | `joints(state)` |

These views are Torch views over Warp arrays. Reading them does not require a
host copy, and writing them updates the simulation.

```python
rod = bodies(state)
positions = rod.positions
rod.linear_velocities = torch.zeros_like(rod.linear_velocities)
```

## NeRD: one learned-dynamics workflow

[Neural Robot Dynamics](https://neural-robot-dynamics.github.io/) learns a
relative physical update from causal state and input history:

```text
delta_state[t] = model(encode(state[t-h+1:t]), input[t-h+1:t])
state[t+1] = integrate(state[t], delta_state[t])
```

The generic workflow separates four responsibilities:

| Piece | Responsibility |
| --- | --- |
| advanced `NeRDStateCodec` | Read/write Newton state and define physical relative updates |
| `NeRDProblem` | Reset and advance one application-specific Newton teacher run |
| `fit_nerd` / advanced `train_nerd` | Collect or consume trajectories, normalize them, and train |
| `TrainedNeRDModel` | Evaluate, save/load, predict, and create a learned `NewtonStepModel` |

The built-in codec factory supports joint, body, particle, and coupled
particle/body state. A custom codec can expose any fixed-topology Newton state.

The common scene path is:

```python
from physicsnemo.experimental.integrations.newton import (
    NeRDControlInput,
    NeRDProblem,
    fit_nerd,
)


control_input = NeRDControlInput("joint_f", per_world_shape=(2,))


def randomize(env, rng):
    # Application-specific initial-state randomization.
    ...


def sample_inputs(rng, world_count, frame):
    # Any global features [worlds, features], or entity-aligned features
    # [worlds, entities, features]. Omit for autonomous dynamics.
    ...


problem = NeRDProblem.from_env(
    env,
    state_codec="joint",
    randomize=randomize,
    sample_inputs=sample_inputs,
    apply_inputs=control_input.apply,
)
trained = fit_nerd(
    problem,
    num_trajectories=10_000,
    steps=100,
    dynamics_model="NeRDTransformer",
)
learned_step = trained.as_step_model(
    newton_model=env.model,
    input_from_step=control_input.from_step,
)
```

The integration intentionally does not prescribe what an input means. It can be
torque, target pose, material features, per-body force, per-particle features,
or any example-defined tensor. `NeRDControlInput` handles one named Newton
control field. `NeRDRigidContactInput` reduces variable-length rigid contacts to
fixed per-body counts, mean normals, and mean body-frame points, while
`concatenate_nerd_inputs` combines global and entity-aligned features. Other
applications can use `NeRDProblem.observe_inputs` to separate controls that
advance the teacher from state-derived model observations. During collection,
`NeRDRigidContactInput.read_env` refreshes contacts before encoding them.
Deployment provides an `input_from_step` callback that returns the exact tensor
shape used for training; returning the Newton `Control` object itself is not
valid. Contact-conditioned deployment must keep collision detection enabled and
derive contacts from the current predicted state rather than replaying recorded
teacher contacts. Because training contacts are usually observed from teacher
states, contact-conditioned autoregressive deployment may also require rollout
or on-policy data to avoid closed-loop distribution shift.

`NeRDBodyStateCodec` makes its coordinate frame explicit. World coordinates are
the default. Use `NeRDFixedFrame` for a fixture, terrain, socket, or other static
environment so the model retains body position relative to collision geometry.
Use `NeRDBodyHeadingFrame` only when translating the body and changing its
heading together with the complete environment leaves the dynamics unchanged.
Removing a non-symmetry makes different physical states indistinguishable and
cannot be repaired by contact features, especially before contact begins.
Transform world-frame vectors and points with `world_vectors_to_model_frame` and
`world_points_to_model_frame` so application inputs use the same coordinates as
the encoded state.

Device placement is also optional: collection follows the Newton state, training
follows the data, and an active distributed run uses its rank-local device.
Pass `device=...` only to override that placement.

Applications that already have trajectories use the same trainer:

```python
from physicsnemo.experimental.integrations.newton.nerd import NeRDDataset, train_nerd

data = NeRDDataset(states=states, inputs=inputs, codec=codec)
trained = train_nerd(data, dynamics_model="NeRDTransformer")
```

## Model choices

Model selection is explicit:

- use `dynamics_model="NeRDTransformer"` for vector state and the causal
  Transformer architecture used by the NeRD paper;
- use `dynamics_model="NeRDEntityTransformer"` for entity-token state, with
  within-frame entity attention and causal temporal attention;
- use `dynamics_model="FullyConnected"` for Markovian vector dynamics that do
  not require temporal mixing.

The integration directly adapts those PhysicsNeMo models. A ready
`torch.nn.Module` or callable receiving `NeRDModelSpec` supports another
architecture without guessing its constructor arguments. The workflow never
selects an architecture from the state shape.

```python
trained = fit_nerd(
    problem,
    num_trajectories=10_000,
    steps=100,
    dynamics_model="FullyConnected",
    model_kwargs={"layer_size": 192, "num_layers": 6},
)
```

The normal forward contract is deliberately small: preserve batch/time and
optional entity dimensions, then return the codec's relative-update shape.

## Deployment

`trained.rollout(...)` free-runs without allocating Newton buffers.
`trained.as_step_model(...)` returns a `NeRDStepModel` implementing the
`NewtonStepModel` contract. It reads and writes live Newton state. One learned
call advances one learned frame, so replacement deployment uses `substeps=1`.

```python
learned_env = NewtonEnv.from_model(
    model,
    step_model=trained.as_step_model(
        newton_model=model,
        input_from_step=control_input.from_step,
    ),
    step_mode="replace",
    collisions=False,
    dt=trained.frame_dt,
    substeps=1,
)
```

Passing the live Newton model lets PhysicsNeMo rebuild the checkpoint's codec
against a compatible model, including a different number of replicated worlds,
and refresh dependent body state after generalized-coordinate predictions.
Custom codecs can be passed explicitly with `state_codec=...`. Keep collision
detection enabled when the learned step or inputs callback consumes live
contacts.

Saved checkpoints include state-layout semantics such as normalized entity and
joint labels, relative index order, topology, and field widths. Loading against
a live Newton model rejects reordered or relabeled entities instead of silently
applying predictions to the wrong physical objects.

## Other workflows

| Goal | Entry point | Example |
| --- | --- | --- |
| Train free-running dynamics with rollout BPTT | `BPTTSurrogate` | [diffsim](../../../../examples/newton/diffsim) |
| Co-design one shared design over task/control candidates | `optimize_grouped_design` | [gripper](../../../../examples/newton/gripper) |
| Optimize when the solver is a black box | `optimize_design`, `DesignSurrogate` | [nozzle](../../../../examples/newton/nozzle) |
| Replace a solver with learned dynamics | `fit_nerd`, `TrainedNeRDModel.as_step_model` | [nerd](../../../../examples/newton/nerd) |

`trajectory_dataset` and `NewtonStepModel` are lower-level boundaries for custom
learning workflows that do not use the high-level NeRD trainer.

## Distributed execution

Every public workflow accepts an explicit device. Under an initialized process
group, NeRD automatically:

- assigns each rank a teacher-trajectory shard;
- computes normalization and active-channel statistics globally;
- divides the configured global batch across ranks; and
- trains with Distributed Data Parallel.

The same API and global configuration work from one GPU to a multi-node job:

```bash
uv run torchrun --standalone --nproc_per_node=<gpus_per_node> train.py
uv run torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> \
  --node_rank=<node_rank> --master_addr=<host> --master_port=<port> train.py
```

Use `resolve_device(args.device)` for rank-local Newton/model placement and
`is_main_process()` to guard one-off reports and checkpoints. Other workflows
remain normal per-process Python unless their documentation states otherwise.

## Installation

Nothing imports Newton at module load time. Install the minimal runtime with the
CUDA backend matching the host, for example
`pip install "nvidia-physicsnemo[cu13,newton]"` (or `cu12`). From a source
checkout, use `uv sync --extra cu13 --extra newton` to run the documented
examples and renderers. The `newton` extra includes Newton's MuJoCo solver
runtime and its example/importer dependencies.
