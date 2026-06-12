Newton Integration Examples
===========================

These examples couple two complementary libraries:

* Newton is the GPU-accelerated physics engine. It builds the physical world,
  detects contacts, and advances positions and velocities with a selected
  numerical solver.
* PhysicsNeMo is the PyTorch-native physics-AI framework. It supplies the model
  zoo, data pipelines, distributed training, learned-physics workflows, and
  deployment adapters.

Start with the :doc:`Newton integration overview
<api/physicsnemo.experimental.integrations.newton>` if either library is new to you. It
explains Newton's model/state/control/solver objects, what PhysicsNeMo provides
over a raw PyTorch training script, and includes a complete first rollout; the
:doc:`examples overview <examples/newton/README>` adds a small first training
loop.

Then choose an example by the question you want to answer:

.. list-table::
   :header-rows: 1

   * - Question
     - Start with
   * - How do I train a surrogate that stays accurate over free-running rollouts?
     - The diffsim cart-pole example
   * - How do I optimize a design when the solver is a black box?
     - The articulated gripper or MPM nozzle example
   * - How do I learn a Newton solver replacement?
     - The NeRD Cartpole and RJ45 cable examples

Each example page explains the physical problem first, then the learning
problem, the PhysicsNeMo API, runnable commands, and the evaluation contract.

.. toctree::
   :maxdepth: 2
   :caption: Examples: Newton
   :name: Examples: Newton

   api/physicsnemo.experimental.integrations.newton.rst
   examples/newton/README.rst
   examples/newton/diffsim/README.rst
   examples/newton/gripper/README.rst
   examples/newton/nozzle/README.rst
   examples/newton/nerd/README.rst
