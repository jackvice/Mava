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

"""Tests for graph observations: the MPE graph wrapper, the batching utilities and the
InforMARL GAT torsos.

These tests are cheap (no training) and check the semantics that the GNN torsos rely on:
per-ego relative node features, visibility-radius topology, and locality of a single
GAT layer.
"""

import importlib
from typing import Dict, List, Tuple

import chex
import jax
import jax.numpy as jnp
import jaxmarl
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from mava.networks.gnn import InforMARLGlobalAggregationTorso, InforMARLNbrhdAggregationTorso
from mava.types import GraphObservation, GraphsTuple, Observation
from mava.utils.graph.gnn_utils import batched_graph_to_single_graph
from mava.wrappers.graph_wrapper import GraphWrapper
from mava.wrappers.jaxmarl import MPEGraphWrapper, MPEWrapper
from test.utils import ConfigValue, find_replace

NUM_AGENTS = 3
NUM_LANDMARKS = 3
NUM_ENTITIES = NUM_AGENTS + NUM_LANDMARKS


def make_mpe_graph_env(visibility_radius: float = 1.0) -> MPEGraphWrapper:
    env = MPEWrapper(
        jaxmarl.make(
            "MPE_simple_spread_v3",
            num_agents=NUM_AGENTS,
            num_landmarks=NUM_LANDMARKS,
            local_ratio=0.5,
        ),
        False,
    )
    return MPEGraphWrapper(env, visibility_radius=visibility_radius)


# --------------------------------------------------------------------------------------
# Wrapper semantics
# --------------------------------------------------------------------------------------


def test_observation_matches_spec() -> None:
    """The graph the wrapper emits must have the shapes/dtypes its spec advertises."""
    env = make_mpe_graph_env()
    spec = env.observation_spec
    _, timestep = env.reset(jax.random.PRNGKey(0))

    assert isinstance(timestep.observation, GraphObservation)
    graph = timestep.observation.graph

    for name in ("nodes", "edges", "senders", "receivers", "n_node", "n_edge", "ego_node_index"):
        expected = getattr(spec.graph, name)
        actual = getattr(graph, name)
        assert actual.shape == expected.shape, f"{name}: {actual.shape} != {expected.shape}"
        assert actual.dtype == expected.dtype, f"{name}: {actual.dtype} != {expected.dtype}"


def test_one_graph_per_agent_with_distinct_ego_index() -> None:
    env = make_mpe_graph_env()
    _, timestep = env.reset(jax.random.PRNGKey(0))
    graph = timestep.observation.graph

    assert graph.nodes.shape == (NUM_AGENTS, NUM_ENTITIES, env.node_features_dim)
    chex.assert_trees_all_equal(graph.ego_node_index.ravel(), jnp.arange(NUM_AGENTS))
    chex.assert_trees_all_equal(graph.n_node.ravel(), jnp.full((NUM_AGENTS,), NUM_ENTITIES))


def test_node_features_carry_entity_type() -> None:
    """Node features must end with an entity type so the GNN can embed it.

    The relative goal position from the paper is omitted because the simple_spread
    scenarios are coverage tasks with no per-agent goal assignment.
    """
    env = make_mpe_graph_env()
    assert env.node_features_dim == 5
    assert env.num_entity_types == 2

    _, timestep = env.reset(jax.random.PRNGKey(0))
    entity_types = timestep.observation.graph.nodes[..., -1]

    # agents come first, then landmarks
    expected = (jnp.arange(NUM_ENTITIES) >= NUM_AGENTS).astype(jnp.float32)
    for ego in range(NUM_AGENTS):
        chex.assert_trees_all_equal(entity_types[ego], expected)


def test_node_features_are_relative_to_ego() -> None:
    """Node features are [rel_pos, rel_vel] w.r.t. the ego agent, so the ego's own row is 0."""
    env = make_mpe_graph_env()
    _, timestep = env.reset(jax.random.PRNGKey(0))
    nodes = timestep.observation.graph.nodes

    ego_rows = nodes[jnp.arange(NUM_AGENTS), jnp.arange(NUM_AGENTS)]
    chex.assert_trees_all_close(ego_rows, jnp.zeros_like(ego_rows))

    # Two different egos must see genuinely different features (this is the whole reason
    # each agent gets its own graph).
    assert not jnp.allclose(nodes[0], nodes[1])


def test_visibility_radius_controls_topology() -> None:
    """A smaller radius must not add edges, and a large radius must fully connect."""
    key = jax.random.PRNGKey(0)

    def num_edges(radius: float) -> int:
        env = make_mpe_graph_env(radius)
        _, timestep = env.reset(key)
        return int((timestep.observation.graph.senders >= 0).sum())

    tight, loose, huge = num_edges(0.2), num_edges(1.0), num_edges(1e6)

    assert tight <= loose <= huge
    assert tight < huge, "radius had no effect on the topology"
    # Self loops are off by default (paper Table 4), so a fully connected graph has
    # num_entities * (num_entities - 1) edges.
    fully_connected = NUM_AGENTS * NUM_ENTITIES * (NUM_ENTITIES - 1)
    assert huge == fully_connected, "large radius should fully connect every graph"


def test_adjacency_is_symmetric_and_respects_radius() -> None:
    """Edges come from a symmetric distance mask, so the adjacency is undirected in effect.

    Note the paper (section 3.1) specifies agent-agent edges as bidirectional but
    agent-to-non-agent edges as unidirectional. Neither Mava nor the reference
    implementation does this: both threshold a symmetric distance matrix, so landmarks
    also receive messages. This test pins the actual (symmetric) behaviour.
    """
    radius = 0.5
    env = make_mpe_graph_env(radius)
    _, timestep = env.reset(jax.random.PRNGKey(0))
    graph = timestep.observation.graph

    for ego in range(NUM_AGENTS):
        senders, receivers = graph.senders[ego], graph.receivers[ego]
        valid = senders >= 0
        adjacency = jnp.zeros((NUM_ENTITIES, NUM_ENTITIES), dtype=bool)
        adjacency = adjacency.at[senders[valid], receivers[valid]].set(True)

        chex.assert_trees_all_equal(adjacency, adjacency.T)

        # every edge is within the radius, measured via the stored edge feature
        distances = graph.edges[ego, :, 0]
        assert jnp.all(distances[valid] <= radius + 1e-6)


def test_edge_features_are_pairwise_distances() -> None:
    env = make_mpe_graph_env(1.0)
    state, timestep = env.reset(jax.random.PRNGKey(0))
    graph = timestep.observation.graph

    positions = state.state.p_pos
    expected = jnp.linalg.norm(positions[:, None] - positions[None, :], axis=-1)

    for ego in range(NUM_AGENTS):
        senders, receivers = graph.senders[ego], graph.receivers[ego]
        valid = senders >= 0
        chex.assert_trees_all_close(
            graph.edges[ego, valid, 0],
            expected[senders[valid], receivers[valid]],
            atol=1e-5,
        )


def test_default_wrapper_respects_add_self_loops() -> None:
    base = MPEWrapper(
        jaxmarl.make(
            "MPE_simple_spread_v3",
            num_agents=NUM_AGENTS,
            num_landmarks=NUM_LANDMARKS,
            local_ratio=0.5,
        ),
        False,
    )
    env = GraphWrapper(base, add_self_loops=False)
    _, timestep = env.reset(jax.random.PRNGKey(0))
    graph = timestep.observation.graph

    assert not jnp.any(graph.senders == graph.receivers), "self loops present despite opting out"


def test_step_keeps_graph_observation() -> None:
    env = make_mpe_graph_env()
    state, timestep = env.reset(jax.random.PRNGKey(0))
    actions = jnp.zeros((NUM_AGENTS, env.action_dim))
    _, next_timestep = env.step(state, actions)

    assert isinstance(next_timestep.observation, GraphObservation)
    chex.assert_trees_all_equal_shapes(
        timestep.observation.graph, next_timestep.observation.graph
    )


# --------------------------------------------------------------------------------------
# Batching utilities
# --------------------------------------------------------------------------------------


def test_batching_offsets_indices_per_graph() -> None:
    """Flattening (T, E, N) graphs into one big graph must keep each sub-graph disjoint."""
    env = make_mpe_graph_env(1e6)  # fully connected -> no padded edges to confuse things
    _, timestep = env.reset(jax.random.PRNGKey(0))

    # add leading time and env dims: (T=2, E=2, N=num_agents, ...)
    graph = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None, None], (2, 2, *x.shape)), timestep.observation.graph
    )
    flat = batched_graph_to_single_graph(graph, num_batch_dims=3)

    num_graphs = 2 * 2 * NUM_AGENTS
    assert flat.n_node.shape == (num_graphs,)
    assert flat.nodes.shape == (num_graphs * NUM_ENTITIES, env.node_features_dim)

    owner = jnp.repeat(jnp.arange(num_graphs), NUM_ENTITIES)
    assert jnp.all(owner[flat.senders] == owner[flat.receivers]), "edges leaked across graphs"
    assert jnp.all(owner[flat.ego_node_index] == jnp.arange(num_graphs))


def test_padded_edges_do_not_become_real_edges() -> None:
    """Padding sentinels must survive the node-index offsetting done when batching."""
    env = make_mpe_graph_env(0.3)  # small radius -> many padded (-1) edges
    _, timestep = env.reset(jax.random.PRNGKey(0))
    graph = timestep.observation.graph

    padded = graph.senders.reshape(-1) < 0
    assert padded.any(), "test needs a radius that actually produces padded edges"

    batched = jax.tree.map(lambda x: x[None, None], graph)
    flat = batched_graph_to_single_graph(batched, num_batch_dims=3)

    assert jnp.all(flat.senders[padded] < 0), "padded senders now index a real node"
    assert jnp.all(flat.receivers[padded] < 0), "padded receivers now index a real node"


# --------------------------------------------------------------------------------------
# GNN torsos
# --------------------------------------------------------------------------------------


def make_synthetic_graph_obs(
    nodes: chex.Array, senders: chex.Array, receivers: chex.Array, ego: int, obs_dim: int = 5
) -> GraphObservation:
    """Builds a (T=1, E=1, N=1) GraphObservation around a single hand-written graph."""
    num_nodes, num_edges = nodes.shape[0], senders.shape[0]
    batch = lambda x: x[None, None, None]

    graph = GraphsTuple(
        nodes=batch(nodes),
        edges=batch(jnp.ones((num_edges, 1))),
        senders=batch(senders),
        receivers=batch(receivers),
        globals=None,
        n_node=batch(jnp.array([num_nodes])),
        n_edge=batch(jnp.array([num_edges])),
        ego_node_index=batch(jnp.array([ego])),
    )
    observation = Observation(
        agents_view=jnp.zeros((1, 1, 1, obs_dim)),
        action_mask=jnp.ones((1, 1, 1, 2), dtype=bool),
        step_count=jnp.zeros((1, 1, 1), dtype=jnp.int32),
    )
    return GraphObservation(observation=observation, graph=graph)


FEATURES_PER_HEAD = 8
NUM_HEADS = 2
TORSO_KWARGS = dict(
    attention_query_layer_sizes=[FEATURES_PER_HEAD],
    use_layer_norm=False,
    activation="relu",
    num_heads=NUM_HEADS,
    num_attention_layers=1,
    concat_heads=False,
)


@pytest.mark.parametrize("concat_heads", [False, True])
def test_neighbourhood_torso_output_shape(concat_heads: bool) -> None:
    nodes = jnp.arange(16.0).reshape(4, 4)
    senders = jnp.array([0, 1, 2, 3])
    receivers = jnp.array([0, 0, 2, 2])
    graph_obs = make_synthetic_graph_obs(nodes, senders, receivers, ego=0)

    torso = InforMARLNbrhdAggregationTorso(**{**TORSO_KWARGS, "concat_heads": concat_heads})
    params = torso.init(jax.random.PRNGKey(0), graph_obs)
    out = torso.apply(params, graph_obs)

    expected = FEATURES_PER_HEAD * (NUM_HEADS if concat_heads else 1)
    assert out.shape == (1, 1, 1, 5 + expected)
    assert jnp.all(jnp.isfinite(out))


def test_global_torso_output_shape() -> None:
    nodes = jnp.arange(16.0).reshape(4, 4)
    senders = jnp.array([0, 1, 2, 3])
    receivers = jnp.array([0, 0, 2, 2])
    graph_obs = make_synthetic_graph_obs(nodes, senders, receivers, ego=0)

    torso = InforMARLGlobalAggregationTorso(**TORSO_KWARGS)
    params = torso.init(jax.random.PRNGKey(0), graph_obs)
    out = torso.apply(params, graph_obs)

    assert out.shape == (1, 1, 1, FEATURES_PER_HEAD)
    assert jnp.all(jnp.isfinite(out))


def test_entity_embedding_stage_distinguishes_types() -> None:
    """With num_entity_types set, the last node feature column is an embedded type."""
    # node 0 is the ego and receives from its single neighbour, node 1
    senders = jnp.array([1])
    receivers = jnp.array([0])
    nodes = jnp.array([[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 0.0]], dtype=jnp.float32)

    torso = InforMARLNbrhdAggregationTorso(
        **{
            **TORSO_KWARGS,
            "num_entity_types": 2,
            "entity_embed_layer_sizes": [FEATURES_PER_HEAD],
        }
    )
    graph_obs = make_synthetic_graph_obs(nodes, senders, receivers, ego=0)
    params = torso.init(jax.random.PRNGKey(0), graph_obs)

    baseline = torso.apply(params, graph_obs)
    # changing only the neighbour's entity type must change the ego embedding
    retyped = nodes.at[1, -1].set(1.0)
    changed = torso.apply(params, make_synthetic_graph_obs(retyped, senders, receivers, ego=0))

    assert jnp.all(jnp.isfinite(baseline))
    assert not jnp.allclose(baseline, changed, atol=1e-5)


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_entity_embedding_supports_both_aggregations(aggregation: str) -> None:
    """Both aggregations must run and stay finite, including for a node with no neighbours.

    A mean divides by the neighbour count, so node 3 below (which receives nothing) is the
    case that would produce NaN if the denominator were not clamped.
    """
    senders = jnp.array([1, 2, -1])
    receivers = jnp.array([0, 0, -1])
    nodes = jnp.arange(16.0).reshape(4, 4)
    graph_obs = make_synthetic_graph_obs(nodes, senders, receivers, ego=0)

    torso = InforMARLGlobalAggregationTorso(
        **{
            **TORSO_KWARGS,
            "num_entity_types": 2,
            "entity_embed_layer_sizes": [FEATURES_PER_HEAD],
            "entity_embed_aggregation": aggregation,
        }
    )
    params = torso.init(jax.random.PRNGKey(0), graph_obs)
    out = torso.apply(params, graph_obs)
    assert jnp.all(jnp.isfinite(out))

    grads = jax.grad(lambda p: jnp.sum(torso.apply(p, graph_obs)))(params)
    for leaf in jax.tree.leaves(grads):
        assert jnp.all(jnp.isfinite(leaf)), f"{aggregation} aggregation produced NaN gradients"


def test_unknown_entity_embedding_aggregation_is_rejected() -> None:
    nodes = jnp.arange(16.0).reshape(4, 4)
    graph_obs = make_synthetic_graph_obs(nodes, jnp.array([1]), jnp.array([0]), ego=0)

    torso = InforMARLNbrhdAggregationTorso(
        **{
            **TORSO_KWARGS,
            "num_entity_types": 2,
            "entity_embed_layer_sizes": [FEATURES_PER_HEAD],
            "entity_embed_aggregation": "median",
        }
    )
    with pytest.raises(ValueError, match="Unknown entity embedding aggregation"):
        torso.init(jax.random.PRNGKey(0), graph_obs)


def test_single_gat_layer_is_local() -> None:
    """With one attention layer the ego embedding must only depend on its in-neighbours.

    Graph: 0 <- 1 (plus self loop on 0) and 2 <-> 3, ego = 0.
    Perturbing node 3 is invisible to node 0; perturbing node 1 is not.
    """
    nodes = jnp.arange(16.0).reshape(4, 4)
    senders = jnp.array([0, 1, 2, 3])
    receivers = jnp.array([0, 0, 3, 2])

    def embed(node_features: chex.Array, params: chex.ArrayTree) -> chex.Array:
        graph_obs = make_synthetic_graph_obs(node_features, senders, receivers, ego=0)
        return torso.apply(params, graph_obs)

    torso = InforMARLNbrhdAggregationTorso(**TORSO_KWARGS)
    params = torso.init(
        jax.random.PRNGKey(0), make_synthetic_graph_obs(nodes, senders, receivers, ego=0)
    )

    baseline = embed(nodes, params)
    far_perturbed = embed(nodes.at[3].add(100.0), params)
    near_perturbed = embed(nodes.at[1].add(100.0), params)

    chex.assert_trees_all_close(baseline, far_perturbed, atol=1e-5)
    assert not jnp.allclose(baseline, near_perturbed, atol=1e-5)


def test_torso_is_permutation_equivariant_for_ego() -> None:
    """Relabelling non-ego nodes must not change the ego embedding."""
    nodes = jax.random.normal(jax.random.PRNGKey(1), (4, 4))
    senders = jnp.array([0, 1, 2, 3])
    receivers = jnp.array([0, 0, 0, 0])  # every node feeds the ego

    torso = InforMARLNbrhdAggregationTorso(**TORSO_KWARGS)
    base_obs = make_synthetic_graph_obs(nodes, senders, receivers, ego=0)
    params = torso.init(jax.random.PRNGKey(0), base_obs)

    permutation = jnp.array([0, 3, 1, 2])
    inverse = jnp.argsort(permutation)
    permuted_obs = make_synthetic_graph_obs(
        nodes[permutation], inverse[senders], inverse[receivers], ego=int(inverse[0])
    )

    chex.assert_trees_all_close(
        torso.apply(params, base_obs), torso.apply(params, permuted_obs), atol=1e-5
    )


@pytest.mark.parametrize("num_attention_layers", [0, 1, 2, 3])
def test_global_torso_supports_variable_depth(num_attention_layers: int) -> None:
    nodes = jnp.arange(16.0).reshape(4, 4)
    senders = jnp.array([0, 1, 2, 3])
    receivers = jnp.array([0, 0, 2, 2])
    graph_obs = make_synthetic_graph_obs(nodes, senders, receivers, ego=0)

    torso = InforMARLGlobalAggregationTorso(
        **{**TORSO_KWARGS, "num_attention_layers": num_attention_layers}
    )
    params = torso.init(jax.random.PRNGKey(0), graph_obs)
    out = torso.apply(params, graph_obs)

    expected = 4 if num_attention_layers == 0 else FEATURES_PER_HEAD
    assert out.shape == (1, 1, 1, expected)
    assert jnp.all(jnp.isfinite(out))


def test_ego_embedding_depends_on_own_features_when_isolated() -> None:
    """The root term W1 @ x_i must keep a node's own features in its embedding.

    Without it an isolated node embeds to exactly zero regardless of its features.
    """
    # ego node 0 is isolated: the only edges are between nodes 2 and 3.
    senders = jnp.array([2, 3])
    receivers = jnp.array([3, 2])
    nodes = jnp.arange(16.0).reshape(4, 4)

    torso = InforMARLNbrhdAggregationTorso(**TORSO_KWARGS)
    params = torso.init(
        jax.random.PRNGKey(0), make_synthetic_graph_obs(nodes, senders, receivers, ego=0)
    )

    def embed(node_features: chex.Array) -> chex.Array:
        return torso.apply(params, make_synthetic_graph_obs(node_features, senders, receivers, 0))

    baseline = embed(nodes)
    changed_ego = embed(nodes.at[0].add(100.0))

    assert not jnp.allclose(baseline, changed_ego, atol=1e-5)


@pytest.mark.parametrize(
    "torso_cls", [InforMARLNbrhdAggregationTorso, InforMARLGlobalAggregationTorso]
)
def test_stacked_layers_keep_raw_edge_features(torso_cls: type) -> None:
    """Stacking layers must work: every layer sees the original edge features.

    jraph.GraphNetwork writes messages back into `edges`, so the layer has to restore
    them, matching the reference which passes the same `edge_attr` to every conv layer.
    """
    nodes = jnp.arange(16.0).reshape(4, 4)
    senders = jnp.array([0, 1, 2, 3])
    receivers = jnp.array([0, 0, 2, 2])
    graph_obs = make_synthetic_graph_obs(nodes, senders, receivers, ego=0)

    torso = torso_cls(**{**TORSO_KWARGS, "num_attention_layers": 2})
    params = torso.init(jax.random.PRNGKey(0), graph_obs)
    assert jnp.all(jnp.isfinite(torso.apply(params, graph_obs)))


@pytest.mark.parametrize("num_attention_layers", [1, 2])
def test_gradients_are_finite_with_padded_edges(num_attention_layers: int) -> None:
    """Padded edges must not poison gradients.

    The forward pass survives padding because segment ops drop out-of-range ids, but the
    backward pass of a dropped scatter is a gather. If a padded edge's softmax weight is
    non-finite (which happens when the segment it indexes has no real edges) the gradient
    becomes NaN even though the forward value looks fine.
    """
    # 4 nodes, 2 real edges into node 0, 3 padded edges, and node 3 has no incoming edges
    senders = jnp.array([1, 2, -1, -1, -1])
    receivers = jnp.array([0, 0, -1, -1, -1])
    nodes = jnp.arange(16.0).reshape(4, 4)
    graph_obs = make_synthetic_graph_obs(nodes, senders, receivers, ego=0)

    torso = InforMARLNbrhdAggregationTorso(
        **{**TORSO_KWARGS, "num_attention_layers": num_attention_layers}
    )
    params = torso.init(jax.random.PRNGKey(0), graph_obs)

    out = torso.apply(params, graph_obs)
    assert jnp.all(jnp.isfinite(out)), "forward pass produced non-finite values"

    grads = jax.grad(lambda p: jnp.sum(torso.apply(p, graph_obs)))(params)
    leaves = jax.tree.leaves(grads)
    assert leaves, "expected at least one gradient leaf"
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), "gradient contained NaN or inf"


def test_padded_edges_do_not_change_outputs() -> None:
    """Adding padding to a graph must leave the real node embeddings untouched."""
    nodes = jnp.arange(16.0).reshape(4, 4)
    senders, receivers = jnp.array([1, 2]), jnp.array([0, 0])
    padded_senders = jnp.concatenate([senders, -jnp.ones(4, dtype=jnp.int32)])
    padded_receivers = jnp.concatenate([receivers, -jnp.ones(4, dtype=jnp.int32)])

    torso = InforMARLNbrhdAggregationTorso(**TORSO_KWARGS)
    unpadded = make_synthetic_graph_obs(nodes, senders, receivers, ego=0)
    padded = make_synthetic_graph_obs(nodes, padded_senders, padded_receivers, ego=0)
    params = torso.init(jax.random.PRNGKey(0), unpadded)

    chex.assert_trees_all_close(
        torso.apply(params, unpadded), torso.apply(params, padded), atol=1e-5
    )


def test_two_layers_reach_second_order_neighbours() -> None:
    """A key claim of the paper: stacked layers propagate information further.

    Chain 2 -> 1 -> 0 with ego 0. One layer cannot see node 2; two layers must.
    """
    senders = jnp.array([1, 2])
    receivers = jnp.array([0, 1])
    nodes = jnp.arange(16.0).reshape(4, 4)

    def ego_embedding(num_layers: int, node_features: chex.Array) -> chex.Array:
        torso = InforMARLNbrhdAggregationTorso(
            **{**TORSO_KWARGS, "num_attention_layers": num_layers}
        )
        base = make_synthetic_graph_obs(nodes, senders, receivers, ego=0)
        params = torso.init(jax.random.PRNGKey(0), base)
        return torso.apply(params, make_synthetic_graph_obs(node_features, senders, receivers, 0))

    perturbed = nodes.at[2].add(100.0)

    chex.assert_trees_all_close(ego_embedding(1, nodes), ego_embedding(1, perturbed), atol=1e-5)
    assert not jnp.allclose(ego_embedding(2, nodes), ego_embedding(2, perturbed), atol=1e-5)


# --------------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system_path,env_name,extra_overrides",
    [
        ("ppo.anakin.rec_ippo", "mpe", []),
        ("ppo.anakin.rec_mappo", "mpe", []),
        # The default fully-connected GraphWrapper has no entity type in its node features.
        (
            "ppo.anakin.rec_ippo",
            "rware",
            [
                "network.actor_network.pre_torso.num_entity_types=0",
                "network.critic_network.pre_torso.num_entity_types=0",
            ],
        ),
    ],
)
def test_gnn_systems_run(
    fast_config: Dict[str, ConfigValue],
    system_path: str,
    env_name: str,
    extra_overrides: List[str],
) -> None:
    """Smoke test that GNN-based systems train end to end."""
    _, _, system_name = system_path.split(".")
    with initialize(version_base=None, config_path="../mava/configs/default"):
        cfg: DictConfig = compose(
            config_name=system_name,
            overrides=[f"env={env_name}", "network=rnn_graph", *extra_overrides],
        )
        cfg = OmegaConf.create(
            find_replace(OmegaConf.to_container(cfg, resolve=True), fast_config)
        )

    OmegaConf.set_struct(cfg, False)
    for logger in cfg.logger.loggers.values():
        logger.enabled = False

    system = importlib.import_module(f"mava.systems.{system_path}")
    system.run_experiment(cfg)


def test_feedforward_systems_reject_graph_observations() -> None:
    """The torsos hardcode 3 batch dims (T, E, N), so feed-forward systems cannot be used.

    This documents the limitation; remove it if the torsos are made rank-agnostic.
    """
    nodes = jnp.arange(16.0).reshape(4, 4)
    senders = jnp.array([0, 1, 2, 3])
    receivers = jnp.array([0, 0, 2, 2])
    graph_obs = make_synthetic_graph_obs(nodes, senders, receivers, ego=0)
    # drop the time dimension, as a feed-forward system would
    feedforward_obs: Tuple = jax.tree.map(lambda x: x[0], graph_obs)

    torso = InforMARLNbrhdAggregationTorso(**TORSO_KWARGS)
    with pytest.raises(AssertionError, match="number of batch dimensions"):
        torso.init(jax.random.PRNGKey(0), feedforward_obs)
