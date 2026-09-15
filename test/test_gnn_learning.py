# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functional tests for the InforMARL GNN torsos.

`test_graph_observations.py` is structural: it checks shapes, locality, permutation
invariance and that padding does not leak. All of those would still pass if the network
were incapable of learning anything. This module checks that the torso can actually fit
functions of the graph, and that it needs the graph to do so.

Every test here pairs the GNN against a control that is expected to fail. "Loss went
down" on its own is not evidence, because an MLP fits most things; the signal is the gap
between the GNN and a model that provably cannot see the information being asked for.

Three groups, in increasing cost:

* Trainability - can the torso overfit a single batch, and does every parameter receive
  gradient. This catches a subnetwork that is constructed but never used.
* Graph probes - supervised regression onto quantities that are functions of the graph
  (neighbour count, neighbour centroid, typed neighbour balance, two-hop average). Each
  isolates one mechanism, and each has a control that lacks the required information.
* Scale transfer - train a probe with 3 agents and evaluate it at 7 and 10 without
  retraining, which is the supervised analogue of the paper's Table 2 claim. The
  attention layers transfer essentially perfectly; the sum-based entity embedding does
  not, and there is a test pinning each of those facts.

All tests are marked `slow` so that they stay out of the default test run.
"""

from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import jaxmarl
import optax
import pytest
from flax import linen as nn

from mava.networks.gnn import InforMARLNbrhdAggregationTorso
from mava.networks.torsos import MLPTorso
from mava.types import GraphObservation, GraphsTuple, Observation
from mava.wrappers.jaxmarl import MPEGraphWrapper, MPEWrapper

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _release_compiled_executables() -> Iterator[None]:
    """Drops JAX's compilation cache after every test.

    Each probe compiles a fresh training program, and every compiled CPU executable holds
    several hundred mmap'd sections that are retained for as long as the cache lives.
    Measured at roughly 490 new mappings per fit, which walks into the default
    `vm.max_map_count` of 65530 partway through this module: XLA then fails with
    "allocateMappedMemory failed" and the process aborts, on a machine with 112 GB free.
    Clearing between tests holds the count flat at about 2500.
    """
    yield
    jax.clear_caches()

# Mirrors mava/configs/network/rnn_graph.yaml, which follows Tables 4 and 7 of the paper.
TORSO_CONFIG: Dict[str, Any] = {
    "attention_query_layer_sizes": [16],
    "use_layer_norm": False,
    "activation": "relu",
    "num_heads": 3,
    "num_attention_layers": 2,
    "concat_heads": False,
    "num_entity_types": 2,
    "entity_embedding_size": 3,
    "entity_embed_layer_sizes": [16],
}

# The visibility radius the MPE wrapper defaults to. With 3 agents and 3 landmarks this
# gives a mean ego in-degree of 2.4 with a standard deviation of 1.3, so the probe targets
# have something to vary over, and 76% of egos have at least two neighbours, which is what
# makes the attention softmax (and therefore the query projection) load bearing.
VISIBILITY_RADIUS = 1.0

TRAIN_BATCH = 256
TEST_BATCH = 128
TRAIN_STEPS = 1500
LEARNING_RATE = 3e-3


# --------------------------------------------------------------------------------------
# Environments and batches
# --------------------------------------------------------------------------------------


def make_env(num_agents: int = 3, radius: float = VISIBILITY_RADIUS) -> MPEGraphWrapper:
    """A simple_spread environment with an equal number of agents and landmarks."""
    env = MPEWrapper(
        jaxmarl.make(
            "MPE_simple_spread_v3",
            num_agents=num_agents,
            num_landmarks=num_agents,
            local_ratio=0.5,
        ),
        False,
    )
    return MPEGraphWrapper(env, visibility_radius=radius)


class ProbeData(NamedTuple):
    """A batch of graph observations with regression targets.

    `mask` marks the (env, agent) pairs whose target is defined; the neighbour centroid,
    for instance, is meaningless for an ego with no neighbours.
    """

    graph_obs: GraphObservation
    targets: chex.Array
    mask: chex.Array


# A probe takes the raw batched graph and returns a possibly modified graph together with
# its targets and a validity mask. Returning the graph lets a probe ablate node features,
# which is how the type-blind control is built.
Probe = Callable[[GraphsTuple, chex.PRNGKey], Tuple[GraphsTuple, chex.Array, chex.Array]]


def sample_graphs(env: MPEGraphWrapper, key: chex.PRNGKey, batch_size: int) -> Tuple[Any, Any]:
    """Resets `batch_size` independent copies of the environment."""
    states, timesteps = jax.vmap(env.reset)(jax.random.split(key, batch_size))
    return states, timesteps.observation.graph


def ego_only_observation(env: MPEGraphWrapper, states: Any) -> chex.Array:
    """The ego's own position and velocity, and nothing else.

    The full simple_spread observation already contains every landmark and agent relative
    position, so a probe built on it could be solved without the graph. Trimming it to
    `[pos, vel]` is the local-observation setting the paper actually studies, and it makes
    the observation-only control provably unable to answer any of the probe questions.
    """
    positions = states.state.p_pos[:, : env.num_agents]
    velocities = states.state.p_vel[:, : env.num_agents]
    return jnp.concatenate([positions, velocities], axis=-1)


def to_graph_observation(graph: GraphsTuple, agents_view: chex.Array) -> GraphObservation:
    """Adds the leading time dimension the torsos expect, giving (T=1, E=batch, N=agents)."""
    num_envs, num_agents = agents_view.shape[:2]
    observation = Observation(
        agents_view=agents_view[None],
        action_mask=jnp.ones((1, num_envs, num_agents, 2), dtype=bool),
        step_count=jnp.zeros((1, num_envs, num_agents), dtype=jnp.int32),
    )
    return GraphObservation(observation=observation, graph=jax.tree.map(lambda x: x[None], graph))


def make_probe_data(
    env: MPEGraphWrapper, probe: Probe, key: chex.PRNGKey, batch_size: int
) -> ProbeData:
    graph_key, probe_key = jax.random.split(key)
    states, graph = sample_graphs(env, graph_key, batch_size)
    graph, targets, mask = probe(graph, probe_key)

    graph_obs = to_graph_observation(graph, ego_only_observation(env, states))
    return ProbeData(graph_obs=graph_obs, targets=targets[None], mask=mask[None])


# --------------------------------------------------------------------------------------
# Graph quantities used to build targets
# --------------------------------------------------------------------------------------


def in_edge_mask(graph: GraphsTuple) -> chex.Array:
    """Marks the edges that deliver a message to the ego node, excluding padding."""
    ego = graph.ego_node_index[..., 0]
    return (graph.senders >= 0) & (graph.receivers == ego[..., None])


def gather_sender_features(graph: GraphsTuple, features: chex.Array) -> chex.Array:
    """Gathers per-node `features` onto edges by sender index. Padded edges read node 0."""
    safe_senders = jnp.where(graph.senders >= 0, graph.senders, 0)
    return jnp.take_along_axis(features, safe_senders[..., None], axis=2)


def adjacency(graph: GraphsTuple) -> chex.Array:
    """Dense adjacency per graph, where entry [s, r] is 1 when s sends a message to r."""
    num_nodes = graph.nodes.shape[2]
    valid = (graph.senders >= 0).astype(jnp.float32)
    safe_senders = jnp.where(graph.senders >= 0, graph.senders, 0)
    safe_receivers = jnp.where(graph.receivers >= 0, graph.receivers, 0)

    def build(valid: chex.Array, senders: chex.Array, receivers: chex.Array) -> chex.Array:
        return jnp.zeros((num_nodes, num_nodes)).at[senders, receivers].max(valid)

    return jax.vmap(jax.vmap(build))(valid, safe_senders, safe_receivers)


def mean_over_in_neighbours(adjacency_matrix: chex.Array, values: chex.Array) -> chex.Array:
    """One round of mean aggregation: out[i] is the average of `values` over i's senders."""
    totals = jnp.einsum("...sr,...s->...r", adjacency_matrix, values)
    degrees = jnp.sum(adjacency_matrix, axis=-2)
    return totals / jnp.maximum(degrees, 1.0)


# --------------------------------------------------------------------------------------
# The probes
# --------------------------------------------------------------------------------------


def degree_probe(graph: GraphsTuple, key: chex.PRNGKey) -> Tuple[GraphsTuple, Any, Any]:
    """How many neighbours does the ego have?

    Counting requires sum aggregation. Note that a softmax attention layer alone cannot do
    this - its weights are normalised to one, so it produces a weighted average that is
    insensitive to how many terms went into it. The count has to come from
    `EntityEmbedConv`, which aggregates with `segment_sum`.
    """
    degree = jnp.sum(in_edge_mask(graph), axis=-1).astype(jnp.float32)
    return graph, degree[..., None], jnp.ones_like(degree)


def centroid_probe(graph: GraphsTuple, key: chex.PRNGKey) -> Tuple[GraphsTuple, Any, Any]:
    """Where is the centre of mass of the ego's neighbours?

    This is the aggregation an attention layer is built for: a convex combination of
    neighbour features. Node features are already relative to the ego, so the target is
    the mean of the neighbours' first two feature columns.
    """
    mask = in_edge_mask(graph)
    neighbour_positions = gather_sender_features(graph, graph.nodes[..., :2])

    degree = jnp.sum(mask, axis=-1).astype(jnp.float32)
    totals = jnp.sum(neighbour_positions * mask[..., None], axis=-2)
    centroid = totals / jnp.maximum(degree, 1.0)[..., None]

    # An ego with no neighbours has no centroid, so drop it rather than train on a zero.
    return graph, centroid, (degree > 0).astype(jnp.float32)


def typed_balance_probe(graph: GraphsTuple, key: chex.PRNGKey) -> Tuple[GraphsTuple, Any, Any]:
    """How many more agents than landmarks are among the ego's neighbours?

    The difference, rather than a plain count of one type, is deliberate: a type-blind
    model can see the total degree but has no way to split it. This is the only probe that
    depends on the entity type, and when the entity embedding stage is enabled it is also
    the only path by which the type can reach the output, because `EntityEmbedConv`
    replaces the node features (type column included) with its own aggregate.
    """
    mask = in_edge_mask(graph)
    # The wrapper writes 0 for agents and 1 for landmarks in the final feature column.
    is_landmark = gather_sender_features(graph, graph.nodes[..., -1:])[..., 0]

    signed = jnp.where(is_landmark > 0.5, -1.0, 1.0) * mask
    return graph, jnp.sum(signed, axis=-1)[..., None], jnp.ones(mask.shape[:-1])


def two_hop_probe(graph: GraphsTuple, key: chex.PRNGKey) -> Tuple[GraphsTuple, Any, Any]:
    """What is the average tag of the ego's neighbours' neighbours?

    The target is exactly two rounds of mean aggregation, so one round cannot produce it.
    The first node feature column is replaced by an i.i.d. random tag: with the real
    positions there, spatial smoothness would let a one-hop model guess the two-hop answer
    and blunt the comparison.

    The ego is itself a neighbour of its neighbours, so its own tag does contribute to the
    target and a one-hop model can recover part of it through the root term. The claim
    tested is that two layers do substantially better, not that one layer is helpless.
    """
    tags = jax.random.normal(key, graph.nodes.shape[:3])
    graph = graph._replace(nodes=graph.nodes.at[..., 0].set(tags))

    adjacency_matrix = adjacency(graph)
    one_hop = mean_over_in_neighbours(adjacency_matrix, tags)
    two_hop = mean_over_in_neighbours(adjacency_matrix, one_hop)

    ego = graph.ego_node_index[..., 0]
    target = jnp.take_along_axis(two_hop, ego[..., None], axis=-1)

    # Restrict to egos that have a two-hop neighbourhood at all.
    degree = jnp.sum(adjacency_matrix, axis=-2)
    ego_degree = jnp.take_along_axis(degree, ego[..., None], axis=-1)[..., 0]
    return graph, target, (ego_degree > 0).astype(jnp.float32)


def ablate_entity_type(probe: Probe) -> Probe:
    """Wraps a probe so the model sees a constant entity type. Targets are unchanged."""

    def ablated(graph: GraphsTuple, key: chex.PRNGKey) -> Tuple[GraphsTuple, Any, Any]:
        graph, targets, mask = probe(graph, key)
        return graph._replace(nodes=graph.nodes.at[..., -1].set(0.0)), targets, mask

    return ablated


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


class GraphProbeModel(nn.Module):
    """A GNN torso followed by the same readout head every model in this file uses."""

    torso: nn.Module
    output_dim: int
    head_layer_sizes: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, graph_obs: GraphObservation) -> chex.Array:
        embedding = self.torso(graph_obs)
        hidden = MLPTorso(list(self.head_layer_sizes), activation="relu")(embedding)
        return nn.Dense(self.output_dim)(hidden)


class ObservationOnlyModel(nn.Module):
    """Control: an MLP on the ego's own position and velocity, ignoring the graph."""

    output_dim: int
    head_layer_sizes: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, graph_obs: GraphObservation) -> chex.Array:
        hidden = MLPTorso(list(self.head_layer_sizes), activation="relu")(
            graph_obs.observation.agents_view
        )
        return nn.Dense(self.output_dim)(hidden)


class FlatNodesModel(nn.Module):
    """Control: an MLP on every node feature concatenated together.

    This is the baseline the paper's scale-transfer claim is aimed at. It can see
    everything the graph can, but its input width is the node count times the feature
    count, so it cannot even be evaluated on a different number of agents.
    """

    output_dim: int
    head_layer_sizes: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, graph_obs: GraphObservation) -> chex.Array:
        nodes = graph_obs.graph.nodes
        flat = nodes.reshape(*nodes.shape[:3], -1)
        inputs = jnp.concatenate([graph_obs.observation.agents_view, flat], axis=-1)
        hidden = MLPTorso(list(self.head_layer_sizes), activation="relu")(inputs)
        return nn.Dense(self.output_dim)(hidden)


def graph_model(output_dim: int, **overrides: Any) -> GraphProbeModel:
    torso = InforMARLNbrhdAggregationTorso(**{**TORSO_CONFIG, **overrides})
    return GraphProbeModel(torso=torso, output_dim=output_dim)


def no_aggregation_model(output_dim: int) -> GraphProbeModel:
    """Control: the same torso with every message-passing stage switched off.

    With no attention layers and no entity embedding the ego's node features pass straight
    through, and they are relative to the ego, so they are identically zero. Anything this
    model achieves is available without looking at the graph at all.
    """
    return graph_model(output_dim, num_attention_layers=0, num_entity_types=0)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def masked_mse(predictions: chex.Array, data: ProbeData) -> chex.Array:
    squared_error = jnp.sum((predictions - data.targets) ** 2, axis=-1)
    return jnp.sum(squared_error * data.mask) / jnp.sum(data.mask)


def fit(
    model: nn.Module, data: ProbeData, key: chex.PRNGKey, steps: int = TRAIN_STEPS
) -> Tuple[chex.ArrayTree, chex.Array]:
    """Full-batch Adam. Returns the fitted parameters and the loss at every step."""
    params = model.init(key, data.graph_obs)
    optimiser = optax.adam(LEARNING_RATE)

    def loss_fn(params: chex.ArrayTree) -> chex.Array:
        return masked_mse(model.apply(params, data.graph_obs), data)

    def step(
        carry: Tuple[chex.ArrayTree, optax.OptState], _: Any
    ) -> Tuple[Tuple[chex.ArrayTree, optax.OptState], chex.Array]:
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimiser.update(grads, opt_state)
        return (optax.apply_updates(params, updates), opt_state), loss

    (params, _), losses = jax.lax.scan(
        step, (params, optimiser.init(params)), None, length=steps
    )
    return params, losses


class Score(NamedTuple):
    """Held-out fit quality.

    `r2` is the fraction of target variance explained, so zero means "no better than
    always predicting the mean". `mse` is the raw residual, and is what the GNN and its
    controls are compared on, because the interesting quantity is how much of the error a
    control leaves behind that the GNN removes.
    """

    r2: float
    mse: float

    def __repr__(self) -> str:
        return f"R2={self.r2:.3f} mse={self.mse:.4f}"


def score(model: nn.Module, params: chex.ArrayTree, data: ProbeData) -> Score:
    predictions = model.apply(params, data.graph_obs)
    mask = data.mask[..., None]
    count = jnp.sum(mask)
    mean = jnp.sum(data.targets * mask, axis=(0, 1, 2), keepdims=True) / count

    residual = jnp.sum(((predictions - data.targets) ** 2) * mask)
    total = jnp.sum(((data.targets - mean) ** 2) * mask)
    return Score(r2=float(1.0 - residual / total), mse=float(residual / count))


def run_probe(
    model: nn.Module,
    probe: Probe,
    num_agents: int = 3,
    seed: int = 0,
    steps: int = TRAIN_STEPS,
) -> Score:
    """Trains `model` on `probe` and scores it on freshly sampled graphs."""
    env = make_env(num_agents)
    train_key, test_key, init_key = jax.random.split(jax.random.PRNGKey(seed), 3)

    train = make_probe_data(env, probe, train_key, TRAIN_BATCH)
    test = make_probe_data(env, probe, test_key, TEST_BATCH)

    params, _ = fit(model, train, init_key, steps)
    return score(model, params, test)


# --------------------------------------------------------------------------------------
# Rung 1: is the torso trainable at all?
# --------------------------------------------------------------------------------------


def random_target_probe(graph: GraphsTuple, key: chex.PRNGKey) -> Tuple[GraphsTuple, Any, Any]:
    """Targets with no relationship to the graph, for the overfitting test."""
    shape = graph.nodes.shape[:2]
    return graph, jax.random.normal(key, (*shape, 1)), jnp.ones(shape)


def test_torso_can_overfit_a_single_batch() -> None:
    """The lowest bar there is: memorise 64 graphs paired with random targets.

    If the torso cannot drive the loss down on data it is allowed to memorise, no result
    further up the ladder means anything.
    """
    env = make_env()
    data = make_probe_data(env, random_target_probe, jax.random.PRNGKey(0), batch_size=64)

    model = graph_model(output_dim=1)
    _, losses = fit(model, data, jax.random.PRNGKey(1), steps=500)

    assert jnp.isfinite(losses).all(), "training diverged"
    assert losses[-1] < losses[0] / 100, (
        f"loss only fell from {losses[0]:.4f} to {losses[-1]:.4f}; the torso cannot fit "
        "even a memorisable batch"
    )


def test_every_parameter_receives_gradient() -> None:
    """No parameter may be dead.

    The attention layer has four separate projections, a root weight and an entity
    embedding table, all added in the same rewrite. A projection that is constructed and
    then multiplied by zero, or a branch masked out everywhere, would pass every
    structural test in the suite while contributing nothing. It shows up here as a
    gradient of exactly zero.

    The loss is a random linear functional of the output rather than a plain sum, so that
    symmetric contributions cannot cancel and hide a live parameter.
    """
    env = make_env()
    data = make_probe_data(env, random_target_probe, jax.random.PRNGKey(0), batch_size=32)

    model = graph_model(output_dim=4)
    params = model.init(jax.random.PRNGKey(1), data.graph_obs)
    weights = jax.random.normal(jax.random.PRNGKey(2), (4,))

    grads = jax.grad(lambda p: jnp.sum(model.apply(p, data.graph_obs) * weights))(params)

    dead = [
        "/".join(str(k.key) for k in path)
        for path, leaf in jax.tree_util.tree_leaves_with_path(grads)
        if not bool(jnp.any(leaf != 0.0))
    ]
    assert not dead, f"parameters with an identically zero gradient: {dead}"


# --------------------------------------------------------------------------------------
# Rung 2: does the torso actually use the graph?
# --------------------------------------------------------------------------------------

# Each probe is paired with controls that cannot see the information it asks for.
#
# The controls do not sit at exactly zero. MPE positions are drawn uniformly in a box, so
# an ego near the centre tends to have more neighbours and its neighbour centroid tends to
# point inwards; the ego's own position therefore predicts a little of every target. That
# boundary effect is real and a control is entitled to it. What a control cannot do is say
# anything about *which* entities are nearby, so the comparison is made on residual error
# rather than on the control scoring zero.
MAX_ERROR_RATIO = 0.25


def assert_beats_controls(subject: Score, controls: Dict[str, Score], min_r2: float) -> None:
    assert subject.r2 > min_r2, f"the GNN only reached {subject} on a function of its own graph"
    for name, control in controls.items():
        assert subject.mse < MAX_ERROR_RATIO * control.mse, (
            f"the GNN ({subject}) did not clearly beat the {name} control ({control}); "
            "the probe is solvable without the graph"
        )


@pytest.mark.parametrize(
    "probe,output_dim,min_r2,overrides",
    [
        pytest.param(degree_probe, 1, 0.7, {}, id="degree"),
        pytest.param(centroid_probe, 2, 0.7, {}, id="centroid"),
        pytest.param(typed_balance_probe, 1, 0.7, {}, id="typed_balance"),
        # `EntityEmbedConv` is a round of sum aggregation in its own right, and on its own
        # it is enough to solve the probes above - a reversed attention direction still
        # scores 0.99 on the centroid with the stage enabled. Disabling it leaves the
        # attention layers as the only route from a neighbour's features to the ego, which
        # is what makes this case a test of the attention mechanism rather than of message
        # passing in general.
        pytest.param(
            centroid_probe, 2, 0.7, {"num_entity_types": 0}, id="centroid_attention_only"
        ),
    ],
)
def test_gnn_beats_graph_blind_controls(
    probe: Probe, output_dim: int, min_r2: float, overrides: Dict[str, Any]
) -> None:
    """The GNN must fit a function of its graph that neither control can reach."""
    gnn = run_probe(graph_model(output_dim, **overrides), probe)
    controls = {
        "observation-only": run_probe(ObservationOnlyModel(output_dim=output_dim), probe),
        "no-aggregation": run_probe(no_aggregation_model(output_dim), probe),
    }
    print(f"\ngnn={gnn} " + " ".join(f"{k}={v}" for k, v in controls.items()))
    assert_beats_controls(gnn, controls, min_r2)


def test_entity_type_is_required_for_the_typed_probe() -> None:
    """Blanking the entity type must break the typed probe and nothing else.

    This is the control that isolates the entity embedding. The model keeps its full
    architecture and the target is unchanged; only the type column is flattened. A model
    that still scores well is reading the balance out of something other than the type,
    which would mean the probe is not testing what it claims to.
    """
    ablated = run_probe(graph_model(1), ablate_entity_type(typed_balance_probe))
    assert ablated.r2 < 0.2, f"type-blind model reached {ablated} on a typed target"

    # The same ablation must not hurt a probe that does not depend on the type, otherwise
    # the failure above could just be damage from perturbing the inputs.
    unaffected = run_probe(graph_model(1), ablate_entity_type(degree_probe))
    assert unaffected.r2 > 0.7, f"blanking the type also broke the degree probe ({unaffected})"


def test_sum_aggregation_counts_better_than_mean() -> None:
    """The cost of making `EntityEmbedConv` scale-invariant, measured.

    A sum can count its terms and a mean cannot, so switching the entity stage to a mean
    has to give something up. The degree probe measures exactly that. Measured: a sum
    reaches an R^2 of 1.000 and a mean 0.829, against graph-blind controls at 0.27.

    So the loss is real but partial - a mean still beats the controls comfortably, because
    the edge features are distances and the average distance to a neighbour carries some
    information about how many there are at a fixed radius. Worth weighing against the
    transfer results below, where the ordering is reversed and far more dramatic.
    """
    summed = run_probe(graph_model(1, entity_embed_aggregation="sum"), degree_probe)
    averaged = run_probe(graph_model(1, entity_embed_aggregation="mean"), degree_probe)
    print(f"\ndegree probe: sum={summed} mean={averaged}")

    assert summed.r2 > averaged.r2 + 0.05, (
        f"a sum was expected to count better than a mean: sum={summed} mean={averaged}"
    )
    assert averaged.r2 > 0.6, (
        f"a mean lost more counting ability than expected ({averaged}); it should still be "
        "well clear of the graph-blind controls at roughly 0.27"
    )


def test_two_layers_beat_one_on_a_two_hop_target() -> None:
    """Stacking attention layers must buy reach, not just avoid crashing.

    The entity embedding is switched off for this comparison because `EntityEmbedConv` is
    itself a round of message passing: with it enabled a single attention layer would
    already see two hops, and the layer count would not equal the propagation depth.
    """
    one_layer = run_probe(graph_model(1, num_attention_layers=1, num_entity_types=0), two_hop_probe)
    two_layer = run_probe(graph_model(1, num_attention_layers=2, num_entity_types=0), two_hop_probe)
    print(f"\none_layer={one_layer} two_layer={two_layer}")

    assert two_layer.r2 > 0.5, f"two layers only reached {two_layer} on a two-hop target"
    assert two_layer.r2 > one_layer.r2 + 0.2, (
        f"a second layer added nothing: {one_layer} -> {two_layer}"
    )


# --------------------------------------------------------------------------------------
# Rung 3: does it transfer across graph sizes?
# --------------------------------------------------------------------------------------


TRANSFER_SIZES = (3, 7, 10)


def train_at_three_and_evaluate_across_sizes(model: nn.Module) -> Dict[int, Score]:
    """Fits the centroid probe with 3 agents and scores it at every transfer size.

    The centroid is used rather than the degree because it is the only probe target whose
    scale does not change with the number of agents. A model trained to emit degrees in
    [0, 5] could not be expected to emit degrees in [0, 19], and failing that would say
    nothing about the architecture.
    """
    train_key, init_key = jax.random.split(jax.random.PRNGKey(0))
    train = make_probe_data(make_env(3), centroid_probe, train_key, TRAIN_BATCH)
    params, _ = fit(model, train, init_key)

    scores = {}
    for num_agents in TRANSFER_SIZES:
        test = make_probe_data(
            make_env(num_agents), centroid_probe, jax.random.PRNGKey(100 + num_agents), TEST_BATCH
        )
        scores[num_agents] = score(model, params, test)
    return scores


def test_attention_transfers_to_larger_graphs() -> None:
    """Train with 3 agents, evaluate at 7 and 10 without retraining.

    This is the supervised form of the paper's Table 2 claim at a fraction of the cost of
    running it in RL. The attention layers are scale-invariant by construction - the
    softmax normalises over however many neighbours there happen to be - so a target that
    is itself scale-invariant should transfer essentially for free. It does, which
    establishes that nothing else in the torso is quietly size-dependent.
    """
    scores = train_at_three_and_evaluate_across_sizes(graph_model(2, num_entity_types=0))
    print(f"\nattention only: {scores}")

    assert scores[3].r2 > 0.9, f"the probe did not fit at its training size: {scores}"
    for num_agents in (7, 10):
        assert scores[num_agents].r2 > 0.9, f"transfer to {num_agents} agents degraded: {scores}"


def test_sum_based_entity_embedding_does_not_transfer() -> None:
    """A sum in `EntityEmbedConv` does not survive a change of graph size.

    This pins a real limitation rather than a bug. The reference declares `EmbedConv` with
    `aggr='add'` (`InforMARL/onpolicy/algorithms/utils/gnn.py`), so Mava is faithful here,
    but the consequence is that the embedding entering the attention layers grows with the
    neighbour count: mean absolute activation roughly triples from 3 agents to 10.
    Evaluated outside its training size the model is extrapolating, and the centroid probe
    degrades from an R^2 of 0.987 at 3 agents to -0.044 at 10.

    That matters for the paper's headline claim, and it is worth knowing before spending
    GPU hours on the RL version of the experiment.
    """
    scores = train_at_three_and_evaluate_across_sizes(
        graph_model(2, entity_embed_aggregation="sum")
    )
    print(f"\nentity stage, sum: {scores}")

    assert scores[3].r2 > 0.9, f"the probe did not fit at its training size: {scores}"
    assert scores[10].r2 < scores[3].r2 - 0.3, (
        f"a sum unexpectedly transferred to 10 agents; if `EntityEmbedConv` was changed "
        f"this test is obsolete. {scores}"
    )


def test_mean_based_entity_embedding_transfers() -> None:
    """Averaging in `EntityEmbedConv` restores transfer completely.

    Measured R^2 across 3, 7 and 10 agents: 0.999, 0.999, 0.999, against 0.987, 0.734 and
    -0.044 for the sum. Averaging also happens to fit better at the training size itself,
    on this probe and on the two-hop probe, presumably because the neighbour count varies
    within a single graph size too and a sum makes the representation scale with it.

    The cost is counting ability, which `test_sum_aggregation_counts_better_than_mean`
    measures.
    """
    scores = train_at_three_and_evaluate_across_sizes(
        graph_model(2, entity_embed_aggregation="mean")
    )
    print(f"\nentity stage, mean: {scores}")

    for num_agents in TRANSFER_SIZES:
        assert scores[num_agents].r2 > 0.9, (
            f"averaging was expected to transfer across graph sizes: {scores}"
        )


def test_flat_mlp_baseline_cannot_change_graph_size() -> None:
    """The control the transfer claim is aimed at cannot even be evaluated at a new size.

    Concatenating every node feature fixes the input width at (num nodes x feature dim),
    so the trained parameters are meaningless once the node count changes. The GNN is
    indifferent because it consumes one node at a time.
    """
    model = FlatNodesModel(output_dim=2)
    small = make_probe_data(make_env(3), centroid_probe, jax.random.PRNGKey(0), batch_size=8)
    large = make_probe_data(make_env(7), centroid_probe, jax.random.PRNGKey(1), batch_size=8)

    params = model.init(jax.random.PRNGKey(2), small.graph_obs)
    with pytest.raises(Exception, match="(?i)dot_general|shape|dimension"):
        model.apply(params, large.graph_obs)


def test_gnn_parameters_are_independent_of_graph_size() -> None:
    """The same parameter tree must run unchanged on 3, 7 and 10 agent graphs."""
    model = graph_model(output_dim=2)
    small = make_probe_data(make_env(3), centroid_probe, jax.random.PRNGKey(0), batch_size=8)
    params = model.init(jax.random.PRNGKey(1), small.graph_obs)

    shapes: List[Tuple[int, ...]] = []
    for num_agents in (3, 7, 10):
        data = make_probe_data(
            make_env(num_agents), centroid_probe, jax.random.PRNGKey(num_agents), batch_size=8
        )
        output = model.apply(params, data.graph_obs)
        assert jnp.all(jnp.isfinite(output))
        shapes.append(output.shape)

    assert [s[-1] for s in shapes] == [2, 2, 2]
