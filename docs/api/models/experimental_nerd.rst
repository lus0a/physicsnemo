Experimental Neural Robot Dynamics (NeRD)
=========================================

These models live under ``physicsnemo.experimental`` and may change as the
NeRD workflows mature.

``NeRDTransformer`` is a causal relative-dynamics sequence model. It consumes a
history window of per-step input tokens (a robot state embedding concatenated
with its actuation and any conditioning) and emits a per-token prediction.
``forward`` returns all tokens for teacher-forced training; autoregressive
deployment uses the final token. The model is application neutral; the
delta-to-next-state conversion, explicit reference framing, and normalization live
in the caller, such as the Newton NeRD integration.

The transformer reuses ``physicsnemo.nn.TimmSelfAttention``'s native causal
attention path and PhysicsNeMo ``Module`` checkpointing.
The Newton integration supplies high-level teacher harvesting, automatic DDP
training under ``torchrun``, evaluation, and deployment. Select
``dynamics_model="NeRDTransformer"`` explicitly for the paper-compatible vector
architecture; the workflow can also train another compatible PhysicsNeMo or
PyTorch module.

.. autoclass:: physicsnemo.experimental.models.nerd.NeRDTransformer
    :show-inheritance:
    :members:
    :exclude-members: forward

Entity-aware NeRD transformer
-----------------------------

``NeRDEntityTransformer`` retains one token per body, particle, or other entity.
It mixes entities within each frame, then applies causal attention through time.
Select ``dynamics_model="NeRDEntityTransformer"`` explicitly for entity-shaped
state; the workflow does not infer an architecture from state shape.

.. autoclass:: physicsnemo.experimental.models.nerd.NeRDEntityTransformer
    :show-inheritance:
    :members:
    :exclude-members: forward
