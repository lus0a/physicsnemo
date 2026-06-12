.. _physicsnemo-newton-integration:

Experimental PhysicsNeMo Newton Integration
===========================================

.. currentmodule:: physicsnemo.experimental.integrations.newton

This optional integration connects `Newton
<https://github.com/newton-physics/newton>`_, a GPU-accelerated physics engine built
on Warp, to PhysicsNeMo's models and physics-AI workflows.

The API lives under ``physicsnemo.experimental`` and may change as the
integration matures.

Newton owns the physical world and numerical time stepping. PhysicsNeMo owns the
learned models, training, evaluation, distributed execution, and deployment
paths. Supported live Newton fields can be exposed as zero-copy Torch views.

Installation
------------

Install the published package with the CUDA backend matching the host:

.. code-block:: bash

    pip install "nvidia-physicsnemo[cu13,newton]"  # CUDA 13
    pip install "nvidia-physicsnemo[cu12,newton]"  # CUDA 12

Use ``nvidia-physicsnemo[newton]`` when the default PyPI Torch backend is
intended and the PhysicsNeMo CUDA/RAPIDS extras are not needed. This does not
guarantee a CPU-only Torch installation.
The same ``newton`` extra includes Newton's MuJoCo solver runtime plus the
dependencies used by the bundled source examples and renderers.

From a PhysicsNeMo checkout:

.. code-block:: bash

    uv sync --extra cu13 --extra newton

Nothing in ``physicsnemo.experimental.integrations.newton`` imports Newton at module load
time. PhysicsNeMo therefore remains usable without the optional runtime.
Operations that need Newton raise a focused installation error when it is absent.

Newton and PhysicsNeMo
----------------------

The libraries solve different parts of the problem:

.. list-table::
    :header-rows: 1

    * - Layer
      - Responsibility
    * - Newton
      - Build physical worlds, detect contacts, and advance state with a selected
        solver on the configured device
    * - PyTorch
      - Provide tensors, neural-network primitives, autograd, losses, and
        optimizers
    * - PhysicsNeMo
      - Add scientific models, data pipelines, distributed training, and
        reusable physics-AI workflows
    * - This integration
      - Drive Newton, expose live Warp state as Torch tensors, train from it,
        validate against it, and deploy learned steps

A Newton simulation has a few core objects:

.. list-table::
    :header-rows: 1

    * - Newton object
      - Meaning
    * - ``model``
      - Mostly fixed bodies, particles, shapes, masses, joints, and materials
    * - ``state``
      - Positions, orientations, and velocities at one instant
    * - ``control``
      - Actuator targets, forces, or other inputs applied while advancing the
        world
    * - collision pipeline and ``contacts``
      - Geometric interactions found for the current state
    * - ``solver``
      - Numerical method that advances ``state_in`` to ``state_out``

A displayed frame may contain several smaller solver substeps. More substeps can
improve numerical stability and contact resolution, but cost more.
:class:`NewtonEnv` owns this lifecycle while still calling the Newton solver and
collision pipeline selected by the application.

When built with :meth:`NewtonEnv.from_scene`, reset starts from the scene's
authored ``state_0`` or first ``states`` entry. When the scene exposes a
collision pipeline, contacts are refreshed before every solver substep by
default. Pass ``collide_each_substep=False`` only when a scene intentionally
authors reusable one-shot contacts.

Why this goes beyond plain PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PhysicsNeMo models are normal ``torch.nn.Module`` objects. The added value
is the tested physics-AI layer around them:

.. list-table::
    :header-rows: 1

    * - Need
      - Public support
    * - Headless Newton reset, step, substeps, collision, and rollout
      - :class:`NewtonEnv` and :func:`load_example_scene`
    * - Zero-copy live Newton/Warp state access
      - :func:`field_to_torch`, :func:`particles`, :func:`bodies`, and
        :func:`joints`
    * - Differentiable Newton trajectories and Warp adjoints
      - :func:`differentiable_rollout`
    * - Free-running surrogate training with rollout BPTT
      - :class:`BPTTSurrogate`
    * - Simulator-as-oracle optimization
      - :class:`DesignSurrogate` and :func:`optimize_design`
    * - Shared-design optimization over task/control candidates
      - :class:`~physicsnemo.experimental.integrations.newton.design_space.DesignSpace`,
        :class:`~physicsnemo.experimental.integrations.newton.design_space.DesignRegularizer`, and
        :func:`optimize_grouped_design`
    * - Candidate-action supervision and simulator shortlisting
      - :func:`grouped_candidate_ranking_loss` and
        :func:`shortlist_grouped_candidates`
    * - Learned Newton step collection, training, evaluation, and deployment
      - :class:`NeRDProblem`, :func:`fit_nerd`, and
        :class:`TrainedNeRDModel`
    * - Distributed learned-dynamics training
      - Automatic trajectory sharding, global statistics, and Distributed Data
        Parallel

Newton remains the teacher and validation reference. A learned model replaces a
solver only when the user explicitly deploys it through :class:`NewtonStepModel`.

Reusable design spaces
----------------------

The solver-independent
:mod:`physicsnemo.experimental.integrations.newton.design_space` module defines
named physical variables, normalized/physical transforms, integer realization,
Sobol sampling, stable schema fingerprints, and composable constraints.
:func:`optimize_grouped_design` accepts a
:class:`~physicsnemo.experimental.integrations.newton.design_space.DesignSpace` directly and stores it in
the returned :class:`GroupedDesignResult`.

Exact symmetry should be represented by sharing one variable while constructing
multiple parts. For designs that permit controlled asymmetry, compose a
:class:`~physicsnemo.experimental.integrations.newton.design_space.SimilarityConstraint` with
:class:`~physicsnemo.experimental.integrations.newton.design_space.DesignRegularizer`.

:meth:`GroupedDesignResult.trajectory_candidates` exposes proposals from
throughout every optimization path, rather than only the final endpoints.
Combine it with
:func:`~physicsnemo.experimental.integrations.newton.design_space.select_diverse_designs` before expensive
verification. Group labels and required anchor indices can retain coverage
across every optimizer trajectory even when global surrogate scores are
imperfect. Then use
:func:`~physicsnemo.experimental.integrations.newton.design_space.select_verified_design` to accept a
proposal only after it beats a measured incumbent.

For problems with controls, poses, or other candidate actions,
:func:`grouped_candidate_ranking_loss` teaches a surrogate to rank candidates
within each task group. :func:`shortlist_grouped_candidates` then selects a
small, ordered set for authoritative simulation. In a sense, this verifies surrogate-proposed actions
instead of performing a hidden exhaustive search.

Reusable rigid geometry
-----------------------

:class:`NewtonRigidObject` describes named compound geometry independently of
an application scene. Parts may be analytic :class:`NewtonPrimitive` shapes or
PhysicsNeMo triangle meshes wrapped by :class:`NewtonMesh`. The same definition
provides an additive mass estimate, bounds, stable fingerprints, area-weighted
point-cloud sampling, and lazy construction of Newton collision shapes.

The mass value is an additive estimate: primitive and closed-mesh part volumes
are summed and multiplied by the object's density. Overlapping compound parts
are therefore counted more than once. :func:`add_rigid_object_shapes` applies
that density to Newton shapes by default and rejects an explicitly supplied
shape configuration with a conflicting density.

.. code-block:: python

    from physicsnemo.experimental.integrations.newton import (
        NewtonPrimitive,
        NewtonRigidObject,
    )

    object_geometry = NewtonRigidObject(
        "compound_part",
        density=420.0,
        parts=(
            NewtonPrimitive("box", (0.04, 0.03, 0.05)),
            NewtonPrimitive("sphere", (0.02,), position=(0.03, 0.0, 0.02)),
        ),
    )
    point_cloud = object_geometry.sample_surface(512, seed=7)

Use :meth:`NewtonRigidObject.from_mesh` with a ``physicsnemo.mesh.Mesh`` loaded
or generated through PhysicsNeMo. Newton
remains optional until :func:`add_rigid_object_shapes` is called.

Your first rollout
------------------

This example loads a stock Newton ball scene headlessly, reads its particle
state as a Torch observation, and advances 60 frames:

.. code-block:: python

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
        # This stock scene authors one-shot plane contacts.
        collide_each_substep=False,
    )
    initial = env.reset()
    rollout = env.rollout(steps=60)

    print(initial.shape)               # torch.Size([6])
    print(rollout.observations.shape)  # torch.Size([61, 6])

Newton builds and advances the world. PhysicsNeMo provides the headless adapter,
lifecycle, live Torch state, and trajectory. The observation remains plain
application code so it can expose exactly the physical values a model needs.

Changing live state is similarly direct:

.. code-block:: python

    env.reset()
    particles(env.state).velocities = torch.tensor(
        [[0.0, 5.0, -5.0]], device=initial.device
    )
    new_rollout = env.rollout(steps=60)

Controls or forces that Newton clears each substep can use
:meth:`NewtonEnv.step`'s ``before_substep`` hook:

.. code-block:: python

    def apply_force(state, control, contacts, dt, substep):
        control.joint_f.fill_(10.0)


    env.step(before_substep=apply_force)

Live state and parallel worlds
------------------------------

Newton packs different physical systems into different arrays:

.. list-table::
    :header-rows: 1

    * - Physics
      - Newton fields
      - Readable view
    * - Particles, soft bodies, and MPM
      - ``particle_q``, ``particle_qd``
      - :func:`particles`
    * - Rigid bodies and rigid-body cables
      - ``body_q``, ``body_qd``
      - :func:`bodies`
    * - Articulations
      - ``joint_q``, ``joint_qd``
      - :func:`joints`

These properties are live Torch views over Warp arrays. Reading does not require
a host copy, and writing changes the simulation:

.. code-block:: python

    rod = bodies(state)
    positions = rod.positions
    rod.linear_velocities = torch.zeros_like(rod.linear_velocities)

Newton can replicate one scene into many worlds and advance them in one solver
call. NeRD state codecs expose those packed arrays as a dense leading
world/trajectory batch. This lets one device generate many teacher trajectories
or evaluate many designs concurrently.

Choose a workflow
-----------------

.. list-table::
    :header-rows: 1

    * - Goal
      - Start with
      - Complete example
    * - Learn free-running dynamics from trajectories
      - :class:`BPTTSurrogate`
      - :doc:`/examples/newton/diffsim/README`
    * - Optimize with a solver that is a black box
      - :func:`optimize_design`, :class:`DesignSurrogate`,
        :func:`optimize_grouped_design`
      - :doc:`/examples/newton/gripper/README`,
        :doc:`/examples/newton/nozzle/README`
    * - Replace a Newton solver step with learned dynamics
      - :class:`NeRDProblem`, :func:`fit_nerd`
      - :doc:`/examples/newton/nerd/README`

The examples own physical choices such as initial inputs, controls, losses,
and task metrics. The integration owns reusable simulation, data, training,
distributed, evaluation, and deployment machinery.

NeRD learned dynamics
---------------------

`Neural Robot Dynamics <https://neural-robot-dynamics.github.io/>`_ (NeRD)
learns a relative physical update from causal state and input history:

.. math::

    \widehat{\Delta \mathbf{s}}_t
    =
    f_\theta\left(
      E(\mathbf{s}_{t-h+1:t}),
      \mathbf{u}_{t-h+1:t}
    \right),
    \qquad
    \widehat{\mathbf{s}}_{t+1}
    =
    \mathbf{s}_t \oplus \widehat{\Delta \mathbf{s}}_t .

The codec defines the state encoding :math:`E`, the relative target, and the
physical integration operator :math:`\oplus`. Joint angles wrap, rigid-body
orientation changes use rotation vectors and normalized quaternions, and
particle state uses position/velocity deltas.

One generic contract supports different Newton problems:

.. list-table::
    :header-rows: 1

    * - Piece
      - Responsibility
    * - :class:`~physicsnemo.experimental.integrations.newton.nerd.NeRDStateCodec`
      - Read/write Newton state and define physical relative updates
    * - :class:`NeRDProblem`
      - Reset and advance one application-specific Newton teacher run
    * - :func:`fit_nerd` /
        :func:`~physicsnemo.experimental.integrations.newton.nerd.train_nerd`
      - Collect or consume trajectories, normalize them, and train
    * - :class:`TrainedNeRDModel`
      - Evaluate, save/load, predict, and create a learned :class:`NewtonStepModel`

Starting a new problem
~~~~~~~~~~~~~~~~~~~~~~

For a normal Newton scene, users provide only the application-specific reset and
inputs:

.. code-block:: python

    from physicsnemo.experimental.integrations.newton import (
        NeRDControlInput,
        NeRDProblem,
        fit_nerd,
    )


    control_input = NeRDControlInput("joint_f", per_world_shape=(2,))


    def randomize(env, rng):
        # Randomize the initial Newton state over the intended operating range.
        ...


    def sample_inputs(rng, world_count, frame):
        # Return global [worlds, features] or entity-aligned
        # [worlds, entities, features] input. Omit for autonomous dynamics.
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

Inputs are deliberately application-defined. They may be torques, target poses,
material properties, commands, per-body forces, per-particle features, or any
other tensor. A state-feedback policy may be captured by the callback.
:class:`NeRDControlInput` provides matched collection and deployment callbacks
for a named Newton control field. :class:`NeRDRigidContactInput` converts
variable-length rigid contacts to fixed per-body contact counts, mean normals,
and mean body-frame contact points. :func:`concatenate_nerd_inputs` broadcasts
global features and combines them with entity-aligned features.
``NeRDProblem.observe_inputs`` separates sampled controls that advance the
teacher from state-derived features stored for training, and
``NeRDRigidContactInput.read_env`` refreshes live contacts before encoding them.
A custom ``input_from_step`` callback must return the same tensor shape used
during training; returning the Newton ``Control`` object is not valid.
Contact-conditioned deployment must keep collision detection enabled and
recompute contacts from the current predicted state. Because training contacts
are usually observed from teacher states, contact-conditioned autoregressive
deployment may also require rollout or on-policy data to avoid closed-loop
distribution shift. Passing no sampler trains autonomous dynamics.

The advanced :func:`~physicsnemo.experimental.integrations.newton.nerd.nerd_state_codec`
factory requires an explicit joint, body, particle, or composite
representation. A custom
:class:`~physicsnemo.experimental.integrations.newton.nerd.NeRDStateCodec` can
expose any fixed-topology Newton state and define its physically correct
relative update.

For maximal-coordinate floating systems,
:class:`~physicsnemo.experimental.integrations.newton.nerd.NeRDBodyStateCodec`
makes its frame explicit. World coordinates are the default.
:class:`~physicsnemo.experimental.integrations.newton.nerd.NeRDFixedFrame`
preserves body position relative to static fixtures, terrain, or collision
geometry. :class:`~physicsnemo.experimental.integrations.newton.nerd.NeRDBodyHeadingFrame`
removes translation and heading only when moving the body and complete
environment together leaves the dynamics unchanged. A moving frame must not
erase position relative to unrepresented fixed geometry; contacts cannot recover
that information before contact begins. Application-defined inputs are not
transformed automatically. Use ``world_vectors_to_model_frame`` for vectors and
``world_points_to_model_frame`` for points that must share the state frame.
Particle components of a composite codec remain world-frame unless an
application supplies a custom coupled codec. Composite components must declare
disjoint Newton state fields; joint and body codecs cannot be combined because
joint finalization recomputes body state through forward kinematics.

Runs with custom controllers, multiple solvers, or unusual frame logic can
construct ``NeRDProblem(codec, get_state, reset, advance, sample_inputs)``
directly. Applications with existing trajectories can use the advanced dataset
and trainer directly:

.. code-block:: python

    from physicsnemo.experimental.integrations.newton.nerd import (
        NeRDDataset,
        train_nerd,
    )

    data = NeRDDataset(states=states, inputs=inputs, codec=codec)
    trained = train_nerd(data, dynamics_model="NeRDTransformer")

This is the flexibility boundary: the example owns physical meaning; the
library handles shape validation, normalization, causal windows, distributed
training, checkpointing, evaluation, and deployment.

Device placement is inferred as well. Collection follows the Newton state,
training follows the trajectory data, and an active distributed run uses its
rank-local device. Pass ``device=...`` only when the application wants to
override that placement.

Choosing a model
~~~~~~~~~~~~~~~~

Model selection is explicit. Choose
``dynamics_model="NeRDTransformer"`` for vector state and the causal Transformer
architecture used by the NeRD paper. Choose
``dynamics_model="NeRDEntityTransformer"`` for entity-token state, with
within-frame entity attention followed by causal temporal attention. Choose
``dynamics_model="FullyConnected"`` for Markovian vector dynamics that do not
require temporal mixing. The workflow does not infer an architecture from the
state shape.

The workflow directly adapts the ``NeRDTransformer``,
``NeRDEntityTransformer``, and ``FullyConnected`` PhysicsNeMo registry models.
It also accepts a ready ``torch.nn.Module`` or a callable receiving
:class:`~physicsnemo.experimental.integrations.newton.nerd.NeRDModelSpec`. Use a callable for another architecture so its
constructor mapping stays explicit. For example, a Markovian vector-state
problem can use PhysicsNeMo's standard MLP:

.. code-block:: python

    trained = fit_nerd(
        problem,
        num_trajectories=10_000,
        steps=100,
        dynamics_model="FullyConnected",
        model_kwargs={
            "layer_size": 192,
            "num_layers": 6,
            "activation_fn": "silu",
        },
    )

A compatible model preserves the leading batch/time and optional entity
dimensions and returns the codec's relative-update shape. Collection,
normalization, DDP, evaluation, checkpointing, and deployment do not change when
the architecture changes.

Evaluation and deployment
~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`TrainedNeRDModel.evaluate` free-runs from held-out initial state and
compares the learned rollout against Newton. Applications can supply physical
metrics for mixed units or task-specific quantities.

``trained.rollout(...)`` evaluates without allocating Newton buffers.
``trained.as_step_model(...)`` returns a live :class:`NeRDStepModel` that
implements the :class:`NewtonStepModel` contract:

.. code-block:: python

    learned_step = trained.as_step_model(
        newton_model=model,
        input_from_step=control_input.from_step,
    )
    learned_env = NewtonEnv.from_model(
        model,
        step_model=learned_step,
        step_mode="replace",
        collisions=False,
        dt=trained.frame_dt,
        substeps=1,
    )

One learned call advances one learned frame, so a replacement deployment
normally uses ``substeps=1``. The live Newton model lets PhysicsNeMo rebuild the
matching codec for any compatible number of replicated worlds and refresh
dependent body state after generalized-coordinate predictions. Advanced custom
codecs can instead be passed with ``state_codec=...``. Keep collision detection
enabled when the learned step or inputs callback consumes live contacts.

.. _physicsnemo-newton-distributed:

Distributed NeRD training
~~~~~~~~~~~~~~~~~~~~~~~~~

Under an initialized process group, :func:`fit_nerd` and
:func:`~physicsnemo.experimental.integrations.newton.nerd.train_nerd`
automatically shard teacher work, compute global normalization and
active-channel statistics, divide the configured global batch across ranks, and
train with Distributed Data Parallel.

Initialize ``physicsnemo.distributed.DistributedManager`` before building
Newton objects, use :func:`resolve_device` for rank-local device placement, and guard
one-off outputs with :func:`is_main_process`.

.. code-block:: bash

    uv run torchrun --standalone --nproc_per_node=<gpus_per_node> train.py
    uv run torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> \
      --node_rank=<node_rank> --master_addr=<host> --master_port=<port> train.py

The public API and global training configuration do not change between one GPU,
one node, and multiple nodes.

API reference
-------------

Start here
~~~~~~~~~~

.. autoclass:: NewtonEnv
    :members:

.. autoclass:: NewtonRollout
    :members:

.. autofunction:: field_to_torch

.. autofunction:: particles

.. autofunction:: bodies

.. autofunction:: joints

.. autofunction:: resolve_device

.. autofunction:: is_main_process

Learning workflows
~~~~~~~~~~~~~~~~~~

.. autoclass:: DifferentiableRollout
    :members:

.. autofunction:: differentiable_rollout

.. autoclass:: TeacherSample
    :members:

.. autoclass:: TeacherBatch
    :members:

.. autofunction:: collect_teacher_batch

.. autoclass:: ResidualDynamics
    :members:

.. autoclass:: BPTTSurrogate
    :members:

.. autoclass:: DesignSurrogate
    :members:

.. autoclass:: DesignResult
    :members:

.. autofunction:: optimize_design

.. autoclass:: GroupedDesignResult
    :members:

.. autofunction:: optimize_grouped_design

.. autofunction:: grouped_candidate_ranking_loss

.. autofunction:: shortlist_grouped_candidates

.. autofunction:: optimize_field_in_newton

.. autofunction:: optimize_field_in_newton_multistart

.. autofunction:: trajectory_dataset

Design-space schemas and selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DesignVariable
    :members:

.. autoclass:: DesignSpace
    :members:

.. autoclass:: SimilarityConstraint
    :members:

.. autoclass:: SmoothnessConstraint
    :members:

.. autoclass:: DesignRegularizer
    :members:

.. autoclass:: VerifiedDesignSelection
    :members:

.. autofunction:: select_diverse_designs

.. autofunction:: select_verified_design

Geometry workflows
~~~~~~~~~~~~~~~~~~

.. autoclass:: NewtonPrimitive
    :members:

.. autoclass:: NewtonMesh
    :members:

.. autoclass:: NewtonRigidObject
    :members:

.. autofunction:: rigid_object_fingerprint

.. autofunction:: add_rigid_object_shapes

Learned Newton steps
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NeRDProblem
    :members:

.. autoclass:: NeRDTrainingConfig
    :members:

.. autofunction:: fit_nerd

.. autoclass:: TrainedNeRDModel
    :members:

.. autoclass:: NeRDControlInput
    :members:

.. autoclass:: NeRDRigidContactInput
    :members:

.. autofunction:: concatenate_nerd_inputs

.. autoclass:: NeRDStepModel
    :members:

.. autoclass:: NewtonStepModel
    :members:

.. autofunction:: write_state_fields

Advanced construction and NeRD APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These APIs are useful when an application already owns Newton objects or
teacher trajectories. Most new users do not need them for a first rollout or a
standard :func:`fit_nerd` workflow.

.. autoclass:: NewtonComponents
    :members:

.. autofunction:: load_example_scene

:class:`~physicsnemo.experimental.integrations.newton.worlds.WorldView` is
imported from the ``...newton.worlds`` submodule and groups a many-world model's
flat per-entity arrays into per-world reductions.

.. autoclass:: physicsnemo.experimental.integrations.newton.worlds.WorldView
    :members:

.. currentmodule:: physicsnemo.experimental.integrations.newton.nerd

.. autoclass:: JointLayout
    :members:

.. autoclass:: NeRDStateCodec
    :members:

.. autoclass:: NeRDJointStateCodec
    :members:

.. autoclass:: NeRDFixedFrame
    :members:

.. autoclass:: NeRDBodyHeadingFrame
    :members:

.. autoclass:: NeRDBodyStateCodec
    :members:

.. autoclass:: NeRDParticleStateCodec
    :members:

.. autoclass:: NeRDCompositeStateCodec
    :members:

.. autofunction:: nerd_state_codec

.. autoclass:: NeRDDataset
    :members:

.. autoclass:: NeRDModelSpec
    :members:

.. autoclass:: NeRDNormalizers
    :members:

.. autofunction:: collect_nerd_trajectories

.. autofunction:: train_nerd

.. autoclass:: NeRDRolloutEvaluation
    :members:

.. autofunction:: evaluate_nerd

Scene visualization helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared helpers used by the bundled Newton example renderers to frame a Z-up
camera, capture frames from Newton's headless GL viewer, and compose them into
an animated GIF. They live in the ``...newton.visualization`` submodule and are
not re-exported from the package root.

.. important::

    The packaged ``newton`` extra requires Newton 1.3 or newer. When running
    against an older Newton source checkout via ``PYTHONPATH``, note that
    releases before 1.2.2 can segfault when ``ViewerGL.get_frame`` reads from a
    CPU viewer on a host where CUDA is available. For those releases,
    :func:`~physicsnemo.experimental.integrations.newton.visualization.capture_frame`
    raises a clear error first; build the model on CUDA or upgrade Newton.

.. currentmodule:: physicsnemo.experimental.integrations.newton.visualization

.. autofunction:: look_at

.. autofunction:: frame_bounding_box

.. autofunction:: aim_camera

.. autofunction:: headless_viewer

.. autofunction:: capture_frame

.. autofunction:: stack_horizontal

.. autofunction:: draw_text

.. autofunction:: save_gif
