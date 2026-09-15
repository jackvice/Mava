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

# Shape conventions:
# T: number of timesteps
# E: number of environments
# N: number of agents
# V: number of nodes per graph
# F: feature dimension

from typing import Optional, Sequence

import chex
import jraph
import jraph._src.models as jraph_models
import jraph._src.utils as jraph_utils
import numpy as np
from flax import linen as nn
from flax.linen.initializers import orthogonal
from jax import numpy as jnp
from jraph import GraphsTuple as JraphGraphsTuple

from mava.networks.torsos import MLPTorso, _parse_activation_fn
from mava.types import GraphObservation
from mava.utils.graph.gnn_utils import GNN, batched_graph_to_single_graph

# Wrappers pad unused edges with a negative sentinel so that jraph's segment operations
# discard them. Discarding keeps the forward pass correct, but the backward pass of a
# dropped scatter is a gather, so padded rows still receive gradient from whichever node
# the sentinel indexes. Messages therefore have to be masked explicitly.
#
# A finite sentinel is used rather than -inf: if a segment contains only padded edges,
# `segment_max` would return -inf and `logits - (-inf)` would be NaN.
_MASKED_LOGIT = -1e30

# How `EntityEmbedConv` combines a node's incoming messages. This is a real trade-off,
# not a free choice:
#
#   "sum"  matches the reference, which declares `EmbedConv` with `aggr='add'`. A sum is
#          the only aggregator here that can count, because the attention layers that
#          follow normalise their weights to one and therefore compute a weighted average
#          that is blind to how many neighbours contributed. Counting is real information
#          (how crowded is it here, how many of my neighbours are agents rather than
#          landmarks), but a sum grows with the neighbour count, so a model trained at one
#          graph size is extrapolating at another.
#   "mean" is scale-invariant and transfers across graph sizes, at the cost of discarding
#          neighbour counts entirely: it reports proportions rather than totals.
#
# `jraph.segment_mean` clamps its denominator to at least one, so a node with no incoming
# edges aggregates to zero rather than NaN. Padded edges carry a negative receiver index
# and are dropped from both the numerator and the denominator.
_ENTITY_EMBED_AGGREGATIONS = {
    "sum": jraph_utils.segment_sum,
    "mean": jraph_utils.segment_mean,
}


def _valid_edge_mask(graph: JraphGraphsTuple) -> Optional[chex.Array]:
    """Returns a per-edge mask that is False for padding, or None if there is no padding."""
    if graph.receivers is None:
        return None
    return graph.receivers >= 0


class EntityEmbedConv(nn.Module):
    """Embeds entity types and performs a first round of message passing.

    This mirrors `EmbedConv` in the reference InforMARL implementation. Node features are
    expected to arrive as `[..., entity_type]`, with the entity type as the final column.
    The type is passed through an embedding layer, concatenated with the neighbour's
    features and the edge feature, and summed over each node's incoming edges:

        x'_i = sum_{j in N(i)} MLP([x_j, Emb(entity_type_j), e_ij])

    Distinguishing entity types matters because the graph mixes agents with landmarks and
    obstacles, which the agent must treat differently.

    `aggregation` selects the sum over incoming edges shown above ("sum", matching the
    reference) or an average over them ("mean"). See `_ENTITY_EMBED_AGGREGATIONS` for what
    that choice costs.
    """

    num_entity_types: int
    embedding_size: int
    layer_sizes: Sequence[int]
    use_layer_norm: bool
    activation: str
    aggregation: str = "sum"

    @nn.compact
    def __call__(self, graph: JraphGraphsTuple) -> JraphGraphsTuple:
        if self.aggregation not in _ENTITY_EMBED_AGGREGATIONS:
            raise ValueError(
                f"Unknown entity embedding aggregation {self.aggregation!r}; "
                f"expected one of {sorted(_ENTITY_EMBED_AGGREGATIONS)}."
            )

        embed = nn.Embed(num_embeddings=self.num_entity_types, features=self.embedding_size)
        mlp = MLPTorso(
            layer_sizes=self.layer_sizes,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            activate_final=True,
            name="entity_embed_mlp",
        )

        valid_edge = _valid_edge_mask(graph)

        def update_edge_fn(
            edges: jraph_models.EdgeFeatures,
            sent_attributes: jraph_models.SenderFeatures,
            received_attributes: jraph_models.ReceiverFeatures,
            graph_globals: jraph_models.Globals,
        ) -> jraph_models.EdgeFeatures:
            neighbour_features = sent_attributes[..., :-1]
            entity_type = sent_attributes[..., -1].astype(jnp.int32)
            entity_type = jnp.clip(entity_type, 0, self.num_entity_types - 1)

            features = [neighbour_features, embed(entity_type)]
            if edges is not None:
                features.append(edges)
            messages = mlp(jnp.concatenate(features, axis=-1))

            if valid_edge is not None:
                messages = jnp.where(valid_edge[:, None], messages, 0.0)
            return messages

        def update_node_fn(
            nodes: jraph_models.NodeFeatures,
            aggregated_sent_attributes: jraph_models.SenderFeatures,
            aggregated_received_attributes: jraph_models.ReceiverFeatures,
            graph_globals: jraph_models.Globals,
        ) -> jraph_models.NodeFeatures:
            return aggregated_received_attributes

        embed_conv = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=None,
            aggregate_edges_for_nodes_fn=_ENTITY_EMBED_AGGREGATIONS[self.aggregation],
        )
        return embed_conv(graph)._replace(edges=graph.edges)


class GraphMultiHeadAttentionLayer(nn.Module):
    """A graph transformer (UniMP) layer.

    Implements the layer update from https://arxiv.org/abs/2009.03509 as used by
    InforMARL, equivalent to `torch_geometric.nn.TransformerConv` with
    `root_weight=True` and `beta=False`:

        x'_i = W1 @ x_i + sum_{j in N(i)} alpha_ij (W2 @ x_j + W5 @ e_ij)

    with attention coefficients given by multi-head dot product attention

        alpha_ij = softmax_j( (W3 @ x_i)^T (W4 @ x_j + W5 @ e_ij) / sqrt(c) )

    where i is the centre node, j ranges over its in-neighbours and c is the per-head
    feature count. Note the edge embedding is added to the neighbour's key and value, not
    to the centre node's query.
    """

    attention_query_layer_sizes: Sequence[int]
    use_layer_norm: bool
    activation: str

    num_heads: int
    concat_heads: bool = False

    @nn.compact
    def __call__(self, graph: JraphGraphsTuple) -> JraphGraphsTuple:
        features_per_head = self.attention_query_layer_sizes[-1]
        total_output_features = features_per_head * self.num_heads
        layer_sizes = [*self.attention_query_layer_sizes[:-1], total_output_features]

        def projection(name: str) -> MLPTorso:
            # `activate_final=False` keeps these as linear projections, matching the
            # weight matrices W2-W5 of the layer update.
            return MLPTorso(
                layer_sizes=layer_sizes,
                use_layer_norm=self.use_layer_norm,
                activation=self.activation,
                activate_final=False,
                name=name,
            )

        query_projection = projection("attention_query_projection")
        key_projection = projection("attention_key_projection")
        value_projection = projection("attention_value_projection")
        edge_projection = projection("attention_edge_projection")

        root_features = total_output_features if self.concat_heads else features_per_head
        root_projection = nn.Dense(
            root_features, kernel_init=orthogonal(np.sqrt(2)), name="root_projection"
        )

        sum_n_node = graph.nodes.shape[0]
        split_heads = lambda x: x.reshape(x.shape[0], self.num_heads, features_per_head)

        valid_edge = _valid_edge_mask(graph)
        # Padded edges must not steer the softmax towards an arbitrary segment, so point
        # them at segment 0 and give them a masked logit that contributes nothing.
        softmax_segments = graph.receivers
        if valid_edge is not None:
            softmax_segments = jnp.where(valid_edge, graph.receivers, 0)

        def compute_messages_fn(
            edges: jraph_models.EdgeFeatures,
            sent_attributes: jraph_models.SenderFeatures,
            received_attributes: jraph_models.ReceiverFeatures,
            graph_globals: jraph_models.Globals,
        ) -> jraph_models.EdgeFeatures:
            """Computes the attention-weighted messages for each edge.

            In jraph, `sent_attributes` holds the neighbour j's features and
            `received_attributes` holds the centre node i's features.
            """
            query = split_heads(query_projection(received_attributes))
            key = split_heads(key_projection(sent_attributes))
            value = split_heads(value_projection(sent_attributes))

            if edges is not None:
                edge_embedding = split_heads(edge_projection(edges))
                key = key + edge_embedding
                value = value + edge_embedding

            logits = jnp.einsum("ehf,ehf->eh", query, key) / jnp.sqrt(features_per_head)
            if valid_edge is not None:
                logits = jnp.where(valid_edge[:, None], logits, _MASKED_LOGIT)

            weights = jraph_utils.segment_softmax(
                logits, segment_ids=softmax_segments, num_segments=sum_n_node
            )
            messages = value * weights[..., None]

            if valid_edge is not None:
                messages = jnp.where(valid_edge[:, None, None], messages, 0.0)
            return messages

        def node_update_fn(
            nodes: jraph_models.NodeFeatures,
            aggregated_sent_attributes: jraph_models.SenderFeatures,
            aggregated_received_attributes: jraph_models.ReceiverFeatures,
            graph_globals: jraph_models.Globals,
        ) -> jraph_models.NodeFeatures:
            """Aggregates heads and adds the root term.

            Ego node communication is unidirectional, so only the aggregated received
            attributes are used.
            """
            if self.concat_heads:
                num_nodes = aggregated_received_attributes.shape[0]
                aggregated = jnp.reshape(aggregated_received_attributes, (num_nodes, -1))
            else:
                aggregated = jnp.mean(aggregated_received_attributes, axis=1)

            return aggregated + root_projection(nodes)

        multi_head_attn_layer = jraph.GraphNetwork(
            update_edge_fn=compute_messages_fn,
            update_node_fn=node_update_fn,
            attention_logit_fn=None,
            attention_reduce_fn=None,
            update_global_fn=None,
            aggregate_edges_for_nodes_fn=jraph_utils.segment_sum,
        )

        # `GraphNetwork` writes the computed messages back into `edges`. Restore the raw
        # edge features so that stacked layers all see the original distances, as the
        # reference implementation does by passing `edge_attr` to every conv layer.
        return multi_head_attn_layer(graph)._replace(edges=graph.edges)


def apply_gnn_trunk(
    graph: JraphGraphsTuple,
    attention_query_layer_sizes: Sequence[int],
    use_layer_norm: bool,
    activation: str,
    num_heads: int,
    num_attention_layers: int,
    concat_heads: bool,
    num_entity_types: int,
    entity_embedding_size: int,
    entity_embed_layer_sizes: Optional[Sequence[int]],
    entity_embed_aggregation: str = "sum",
) -> JraphGraphsTuple:
    """Runs the optional entity embedding stage followed by the attention layers."""
    activation_fn = _parse_activation_fn(activation)

    if num_entity_types > 0:
        if entity_embed_layer_sizes is None:
            raise ValueError("entity_embed_layer_sizes is required when num_entity_types > 0.")
        graph = EntityEmbedConv(
            num_entity_types=num_entity_types,
            embedding_size=entity_embedding_size,
            layer_sizes=entity_embed_layer_sizes,
            use_layer_norm=use_layer_norm,
            activation=activation,
            aggregation=entity_embed_aggregation,
        )(graph)

    for _ in range(num_attention_layers):
        graph = GraphMultiHeadAttentionLayer(
            attention_query_layer_sizes=attention_query_layer_sizes,
            use_layer_norm=use_layer_norm,
            activation=activation,
            num_heads=num_heads,
            concat_heads=concat_heads,
        )(graph)
        graph = graph._replace(nodes=activation_fn(graph.nodes))

    return graph


class InforMARLNbrhdAggregationTorso(GNN):
    """InforMARL Actor Network.
    For more details see: https://arxiv.org/abs/2211.02127

    Each agent has its own graph, where the agent is called the ego-agent. This torso uses
    multi-layer multi-head graph transformer layers to perform local neighborhood
    aggregation, where each node only aggregates information from its direct neighbors
    using edge information.

    For example, in a graph with nodes:
    A - B
    C - D
    where A is the ego-agent:
    - A's and B's node features will be a function of only A's and B's node features
    - C's and D's node features will be a function of only C's and D's node features

    Since A is the ego-agent, only A's node feature will be taken from this computation
    and concatenated with A's observation.
    """

    attention_query_layer_sizes: Sequence[int]
    use_layer_norm: bool
    activation: str

    num_heads: int
    num_attention_layers: int

    concat_heads: bool = False
    # Set num_entity_types > 0 when node features carry an entity type as their final
    # column, as the MPE graph wrapper does.
    num_entity_types: int = 0
    entity_embedding_size: int = 3
    entity_embed_layer_sizes: Optional[Sequence[int]] = None
    entity_embed_aggregation: str = "sum"

    @nn.compact
    def __call__(self, graph_observation: GraphObservation) -> chex.Array:
        observation = graph_observation.observation
        graph = graph_observation.graph
        obs = observation.agents_view
        T, E, N, *_ = graph.nodes_strict.shape
        # one for timesteps, one for envs, one for agents
        graph = batched_graph_to_single_graph(graph, num_batch_dims=3)

        *_graph, ego_node_index = graph
        jraph_graph = JraphGraphsTuple(*_graph)

        jraph_graph = apply_gnn_trunk(
            jraph_graph,
            attention_query_layer_sizes=self.attention_query_layer_sizes,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            num_heads=self.num_heads,
            num_attention_layers=self.num_attention_layers,
            concat_heads=self.concat_heads,
            num_entity_types=self.num_entity_types,
            entity_embedding_size=self.entity_embedding_size,
            entity_embed_layer_sizes=self.entity_embed_layer_sizes,
            entity_embed_aggregation=self.entity_embed_aggregation,
        )

        ego_node_features = get_ego_node_features(jraph_graph, ego_node_index, T, E, N)
        graph_embedding = jnp.concatenate([obs, ego_node_features], axis=-1)

        return graph_embedding


class InforMARLGlobalAggregationTorso(GNN):
    """InforMARL Critic Network.
    For more details see: https://arxiv.org/abs/2211.02127

    Each agent has its own graph, where the agent is called the ego-agent. This torso uses
    multi-layer multi-head graph transformer layers to aggregate node features, where edge
    information is used in the aggregation.

    Unlike the neighborhood aggregation torso, the ego-agent information is not picked
    after the aggregation. All nodes of the ego-agent's graph are averaged together, which
    keeps the critic input size independent of the number of entities. The attention
    layers can be disabled by setting num_attention_layers to 0.
    """

    attention_query_layer_sizes: Sequence[int]
    use_layer_norm: bool
    activation: str

    num_heads: int
    num_attention_layers: int

    concat_heads: bool = False
    num_entity_types: int = 0
    entity_embedding_size: int = 3
    entity_embed_layer_sizes: Optional[Sequence[int]] = None
    entity_embed_aggregation: str = "sum"

    @nn.compact
    def __call__(self, graph_observation: GraphObservation) -> chex.Array:
        graph = graph_observation.graph
        T, E, N, V, *_ = graph.nodes_strict.shape
        # one for timesteps, one for envs, one for agents
        graph = batched_graph_to_single_graph(graph, num_batch_dims=3)

        *_graph, ego_node_index = graph
        jraph_graph = JraphGraphsTuple(*_graph)

        jraph_graph = apply_gnn_trunk(
            jraph_graph,
            attention_query_layer_sizes=self.attention_query_layer_sizes,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            num_heads=self.num_heads,
            num_attention_layers=self.num_attention_layers,
            concat_heads=self.concat_heads,
            num_entity_types=self.num_entity_types,
            entity_embedding_size=self.entity_embedding_size,
            entity_embed_layer_sizes=self.entity_embed_layer_sizes,
            entity_embed_aggregation=self.entity_embed_aggregation,
        )

        node_embedding = jraph_graph.nodes

        node_features = node_embedding.reshape(
            T,
            E,
            N,
            V,
            *node_embedding.shape[1:],
        )
        # There is a graph for a given timestep, env, and agent. We mean pool the node
        # features of each graph, mirroring `global_mean_pool` in the reference.
        pooled_global_features = jnp.mean(node_features, axis=3)

        return pooled_global_features


def get_ego_node_features(
    graph: JraphGraphsTuple, ego_node_index: chex.Array, *num_nodes: Sequence[int]
) -> chex.Array:
    """Returns the ego node features from a graph."""
    return graph.nodes[ego_node_index].reshape(*num_nodes, *graph.nodes.shape[1:])
