# InforMARL GNN: corrections, faithfulness changes, and tests

This document describes the changes made to Mava's graph-observation / GNN feature on the
`cursor/informarl-gnn-fixes-and-tests` branch. The feature implements InforMARL
([arXiv:2211.02127](https://arxiv.org/abs/2211.02127), ICML 2023); the reference
implementation is the authors' [InforMARL repo](https://github.com/nsidn98/InforMARL).

## Context: the feature had no tests

The feature arrived in commit `17ee9968` ("feat: graph-based observations & GNN torsos
(#1177)"), which added 968 insertions across 10 files and **zero files under `test/`**. It was
never added to the CI matrix and is mentioned in neither the README nor `docs/`. A follow-up,
`e701b7e4` ("fix: is gnn based and pin JAX to 0.5.3 (#1190)"), fixed a post-merge bug two
months later, also without a test.

This matters for reading the rest of this document: there was no validated baseline, so none
of the changes below are regressions against previously-working behaviour. They are the first
time the code has been exercised.

## 1. Correctness bugs

### 1.1 Padding sentinel leak when batching graphs

`mava/utils/graph/gnn_utils.py`

Wrappers pad unused edges with `-1` in `senders`/`receivers` so that jraph's segment
operations discard them, which is how edge arrays keep a static shape under JAX. When
`batch()` concatenated sub-graphs it added each sub-graph's node offset unconditionally, so
`-1 + offset` became a *valid* index pointing at the previous sub-graph's last node. Padded
edges therefore turned into real messages, and they crossed the boundary between two
different agents' ego-graphs.

The fix is a helper that leaves negative sentinels alone:

```python
def _offset_node_indices(indices: jax.Array, offset: jax.Array) -> jax.Array:
    """Shifts node indices by `offset`, leaving negative padding sentinels intact."""
    return jnp.where(indices < 0, indices, indices + offset)
```

applied to `senders` and `receivers` but not `ego_node_index`, which is always valid. At
`visibility_radius=0.3` with 3 agents, 48 of 72 padded edges were being promoted to real
edges before this fix.

While fixing this, `batched_graph_to_single_graph` was also rewritten to be vectorised.
It previously built a Python list of sub-graphs and called `batch()` on it, which emits one
concatenate operand per sub-graph — and there are `timesteps * envs * agents` of them, so
tracing and compilation blew up long before the data did. It now computes offsets with a
single `cumsum` and reshapes.

### 1.2 Stacking attention layers crashed

`mava/networks/gnn.py`

`num_attention_layers >= 2` raised:

```
ValueError: Einstein sum subscript 'ehf' does not contain the correct number of
indices for operand 1
```

`jraph.GraphNetwork` writes the per-head messages it computed back into `graph.edges`, so the
second layer saw 3-D edge features where it expected 2-D and broadcasting pushed the einsum
operands to 4-D. Each layer now restores the raw edge features on the way out:

```python
return multi_head_attn_layer(graph)._replace(edges=graph.edges)
```

This was not an edge case. Two layers is what Table 4 of the paper specifies and what the
reference's `gnn_layer_N` default uses, so the configuration the paper describes could not
run at all.

### 1.3 NaN gradients from padded edges

`mava/networks/gnn.py`

After self-loops were turned off (see 1.4), training produced NaN losses on the second
update. The cause is in `jraph.segment_softmax`, which computes `maxs[segment_ids]`: for a
padded edge the segment id is `-1`, which is ordinary negative indexing to the *last* node.
When that node had no incoming edges, `segment_max` returned `-inf`, so
`exp(logits - (-inf)) = inf`. The forward pass survived because `segment_sum` then dropped
the out-of-range id — but the backward pass of a dropped scatter is a gather, which returned
NaN.

This bug was latent in the original code: with self-loops unconditionally on, every node had
at least one incoming edge, so `segment_max` never returned `-inf`.

Padded edges are now masked explicitly rather than relying on segment ops to drop them.
Their logits are set to a finite sentinel, their segment ids are remapped to 0, and their
messages are zeroed:

```python
# A finite sentinel is used rather than -inf: if a segment contains only padded edges,
# `segment_max` would return -inf and `logits - (-inf)` would be NaN.
_MASKED_LOGIT = -1e30
```

### 1.4 `add_self_loops` was silently ignored

`mava/wrappers/graph_wrapper.py`

The generic `GraphWrapper` accepted an `add_self_loops` flag and then always built a
fully-connected graph *with* self-edges, and its `observation_spec` reported an edge count
that assumed self-loops regardless. The flag is now threaded through to
`jraph.get_fully_connected_graph` and the spec computes the matching count:

```python
edges_per_node = self.num_agents if self.add_self_loops else self.num_agents - 1
max_n_edge = self.num_agents * edges_per_node
```

The MPE wrapper's default also flips from `True` to `False`, per Table 4 of the paper — the
attention layer's root term (see 2.1) is what supplies the self contribution.

## 2. Architectural faithfulness

The layer the paper and the reference implement is UniMP / a graph transformer
([Shi et al.](https://arxiv.org/abs/2009.03509)), equivalent to PyTorch Geometric's
`TransformerConv(root_weight=True, beta=False)`:

\[ x'_i = W_1 x_i + \sum_{j \in N(i)} \alpha_{ij}(W_2 x_j + W_5 e_{ij}) \]
\[ \alpha_{ij} = \mathrm{softmax}\!\left(\frac{(W_3 x_i)^\top (W_4 x_j + W_5 e_{ij})}{\sqrt{c}}\right) \]

Mava's version departed from this in five ways.

### 2.1 Missing root weight \(W_1 x_i\)

The node update consumed only the aggregated neighbour messages. Its `node_update_fn` took
`nodes` as an argument and never used it, so a node with no visible neighbours embedded to
*exactly zero* — the architecture could not represent "I see nothing, here is where I am".
A `root_projection` dense layer was added and its output is added to the aggregated heads.

### 2.2 Collapsed Q/K/V/edge projections, with the edge on the wrong side

A single shared MLP produced the query, key, and value, and the edge embedding was added to
the *query* (the centre node i) rather than to the neighbour's key and value. There are now
four separate linear projections — `attention_query_projection`, `attention_key_projection`,
`attention_value_projection`, `attention_edge_projection` — with the query taken from the
centre node, key and value from the neighbour, and the edge embedding added to both the key
and the value. The projections are also linear now (`activate_final=False`); applying a
nonlinearity inside the projection is not what either the paper or the reference does.

### 2.3 Non-uniform head aggregation

Head aggregation was decided per-layer by `should_avg_multi_head = i < num_attention_layers - 1`,
so intermediate layers averaged their heads but the last one concatenated — an arbitrary
asymmetry present in neither source. This is now a single `concat_heads` flag applied
uniformly, defaulting to `False` (average) to match the reference.

### 2.4 Activation placement

The activation was applied inside the projections rather than between layers. It now sits
between attention layers, in `apply_gnn_trunk`.

### 2.5 Missing entity-type embedding

The reference's `EmbedConv` prepends a message-passing stage that embeds a discrete entity
type (agent / landmark / obstacle) and concatenates it to the neighbour features before the
MLP. Mava had no equivalent, so the network could not tell an agent from a landmark. A new
`EntityEmbedConv` module implements

\[ x'_i = \sum_{j \in N(i)} \mathrm{MLP}([x_j, \mathrm{Emb}(\text{type}_j), e_{ij}]) \]

and runs ahead of the attention layers when `num_entity_types > 0`.

The aggregation is selectable via `entity_embed_aggregation` (`"sum"`, matching the
reference, or `"mean"`). This turned out to matter: the learning tests show that the
sum-based embedding **does not transfer to graphs larger than it was trained on**, because
the summed magnitude scales with degree, while the mean-based one does. The attention layers
themselves transfer either way, since they normalise over neighbours.

### 2.6 Two deviations deliberately *not* changed

Two things that look like deviations from the paper's prose actually match the reference
implementation, and were left alone:

- **Symmetric adjacency, including landmark–landmark edges.** The paper's prose implies
  directed edges from non-agents to agents; the reference builds a symmetric adjacency.
- **Per-ego-graph mean pooling for the critic.** `InforMARLGlobalAggregationTorso` pools
  within each ego-graph rather than over a single global graph, which is what the reference's
  `global_mean_pool` does given its batching.

## 3. Environment and wrapper changes

### 3.1 Entity type in MPE node features

`mava/wrappers/jaxmarl.py`

MPE node features go from 4 to 5 dimensions —
`[relative_x, relative_y, relative_vx, relative_vy, entity_type]` — and the wrapper exposes
`num_entity_types = 2`.

The reference also includes each entity's relative goal position. That is omitted here, and
deliberately: the `simple_spread` scenarios are coverage tasks with no per-agent goal
assignment, so there is no goal to report.

### 3.2 Local-observation mode

`mava/wrappers/jaxmarl.py`, `mava/configs/env/mpe.yaml`

This addresses a measurement problem rather than a bug. The MPE observation is

```
[self_vel (2), self_pos (2), landmark_rel_pos, other_agent_rel_pos, comm]
```

which already contains the relative position of every landmark and every other agent — that
is the *global* condition in the paper. Because the graph torso concatenates its embedding
onto that observation, a graph network is strictly better informed than an MLP on the same
input, and "the GNN beats the MLP" becomes close to tautological.

The paper's actual claim is that a GNN with only *local* information matches an MLP with
global information. `MPEWrapper(local_observations=True)` trims the observation to its
leading four columns, so the agent knows only its own velocity and position and the graph
becomes the only source of information about anything else. Only the actor's view is
restricted; `global_state` is left intact, because the critic is centralised in both the
paper and MAPPO.

A side effect worth noting: this makes the actor's input width independent of scenario size
(4 at 3, 5 and 10 agents, against 18, 30 and 60 by default), so a checkpoint can be evaluated
at a size it was not trained on — provided `system.add_agent_id` is `False`, since the
one-hot it prepends is as long as the number of agents.

### 3.3 Wrapper options reachable from Hydra

`mava/utils/make_env.py`

Environment wrappers are constructed in `make_env.py` rather than by Hydra, and the graph
wrapper was instantiated with no arguments at all — so `visibility_radius`, the single most
important knob in the paper, was unreachable from config. Two optional passthrough dicts were
added, read by a new `_wrapper_kwargs` helper: `env.wrapper_kwargs` (forwarded to the base
env wrapper) and `env.graph_wrapper_kwargs` (forwarded to the graph wrapper). Envs that do
not define them keep their wrapper defaults.

Note that at the default `visibility_radius: 1.0` the graph is close to complete — 24 of 36
possible edges with 3 agents — which erases most of the locality the architecture exists to
exploit. Smaller radii are the interesting regime (see Figure 7 of the paper).

### 3.4 New scenario

`mava/configs/env/scenario/simple_spread_15ag_10lm.yaml` adds a 15-agent / 10-landmark
scenario. Unlike the symmetric `simple_spread_{3,5,10}ag` scenarios, agents outnumber
landmarks, so they cannot all occupy a distinct goal.

## 4. Configuration

`mava/configs/network/rnn_graph.yaml` now follows Tables 4 and 7 of the paper:

| Field | Was | Now |
| --- | --- | --- |
| `hidden_state_dim` | 128 | 64 |
| `attention_query_layer_sizes` | `[128]` | `[16]` |
| `num_heads` | 4 | 3 |
| `num_attention_layers` | 1 | 2 |
| `post_torso.layer_sizes` | `[128]` | `[64, 64]` |
| `concat_heads` | — | `False` |
| `num_entity_types` | — | 2 |
| `entity_embedding_size` | — | 3 |
| `entity_embed_layer_sizes` | — | `[16]` |
| `entity_embed_aggregation` | — | `sum` |

`num_entity_types` must match the wrapper: 2 for MPE, 0 for the generic `GraphWrapper`, whose
node features carry no type.

## 5. Tests

Two new files, 59 tests.

**`test/test_graph_observations.py`** (45 tests, fast) covers wrapper semantics (spec and
shape agreement, one graph per agent, ego-relative features, radius topology, adjacency
symmetry, edge features equal to distances, `add_self_loops` respected), the batching
utilities (index offsetting, padded edges staying negative), and the GNN torsos (output
shapes across `concat_heads`, locality of a single layer, permutation equivariance, variable
depth 0–3, finite gradients with padded edges, padding not changing outputs, stacked layers
keeping raw edges, two layers reaching second-order neighbours, the root weight on an
isolated node). Each bug in section 1 has a test that fails without its fix.

**`test/test_gnn_learning.py`** (14 tests, marked `slow`) actually trains the torso on
synthetic graph probes with known answers — node degree, neighbour centroid, typed balance,
two-hop reach — and checks it beats graph-blind controls. This is what established that the
architecture learns the things it is supposed to be able to learn, including that entity type
is *required* for the typed probe, that sum aggregation counts better than mean, that two
layers beat one on a two-hop target, and the size-transfer results in 2.5. A `slow` marker was
registered in `pyproject.toml` so the default run can exclude them with `-m 'not slow'`.

**`test/integration_test.py`** gained `test_graph_system`, which runs `network=rnn_graph` on
`rec_ippo` and `rec_mappo` across `mpe` and `rware`, so CI exercises the feature for the
first time. Only recurrent systems are included: the GNN torsos assume three batch dims
(time, env, agent), so feedforward systems cannot consume graph observations — and there is a
test asserting they reject them.

## 6. Caveats

**The work is architecturally faithful but not benchmark-validated.** The tests establish
that the implementation computes what the paper describes and can learn graph-structured
targets. No learning-curve comparison against the reference implementation on the paper's
benchmarks has been run, so there is no evidence that it reproduces the paper's *results*.

**`visibility_radius` is now reachable but the defaults are not informative.** At radius 1.0
the graph is nearly complete, and with `local_observations: False` the GNN-vs-MLP comparison
is confounded as described in 3.2. Any evaluation should set both deliberately.

## Changed files

Relative to the merge base with `origin/develop`:

```
 mava/configs/env/mpe.yaml                               |  27 +
 mava/configs/env/scenario/simple_spread_15ag_10lm.yaml  |  10 +
 mava/configs/network/rnn_graph.yaml                     |  34 +-
 mava/networks/gnn.py                                    | 480 ++++++++----
 mava/utils/graph/gnn_utils.py                           |  50 +-
 mava/utils/make_env.py                                  |  25 +-
 mava/wrappers/graph_wrapper.py                          |   4 +-
 mava/wrappers/jaxmarl.py                                |  77 +-
 pyproject.toml                                          |   5 +
 test/integration_test.py                                |  27 +
 test/test_gnn_learning.py                               | 754 +++++++++++++++++
 test/test_graph_observations.py                         | 802 +++++++++++++++++
```

Principal commits: `3483dc19` ("fix: correct InforMARL GNN torso and add functional tests")
and `cb3d0b8f` ("feat: add local-observation mode for MPE and plumb wrapper options").
