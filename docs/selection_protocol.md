# Frozen PARC Selection Protocol

This documents the development procedure underlying the released model. It
does not change the selector, rerun experiments, or search for a new policy.
The [machine-readable protocol](../configs/parc_selection_protocol.json) and
[model manifest](../artifacts/parc_selector/parc_frozen_selector_manifest.json)
contain the exact grid order and selected constants.

## The pool precedes final evidence

For a query, each candidate route first constructs its retained chunk pool,
then determines an ordered evidence list inside that pool. Hybrid routes
restrict replacement candidates to retained chunks. XE-based routes score
their retained pools; LM and XE replacement tests normalize XE scores over
the retained pool. Thus a chunk can affect candidate eligibility or score
normalization before evidence is finalized, even if it is not in the final
top-8 list. Pool size and retained-position features also enter routing.

Dense-only views preserve Dense evidence identities and vary order; their
extra retained chunks need not change the answer. They remain useful as
compressed Dense-equivalent anchors within the predefined portfolio. Only
the chosen ordered evidence enters the generator prompt. The evaluated
procedure does not implement updates to a persistent memory across queries,
and pool-count reduction is not a measured end-to-end memory/latency saving.

## Why these nine routes and budgets

The portfolio covers three controlled differences: evidence order,
replacement signal, and pool retention. Three Dense-only views vary order;
BM25, RRF, and XE hybrids vary the replacement signal; three LM variants vary
retention under the same language-conditioned replacement rule.

The 45% nominal retention setting was inherited from the initial LM
configuration documented on 2026-07-16. The development inventory of
2026-07-18 added LM-35 and LM-55 by changing only the name and retention ratio
of LM-45. All nine routes were fixed before final selector training. These
are heuristic predefined settings, not an exhaustive factorial search, a
continuous budget optimum, or a new sensitivity result. Small-pool rounding
and strict-compression bounds can make nominal budgets coincide.

## Raw and augmented predictors

Raw features describe score magnitudes, Dense-evidence overlap, pool size,
and related retrieval observations. Query-relative transformations compare
the candidate routes for the same query. A high within-query rank does not
alone establish strong absolute support. The augmented view retains all 221
raw-view features and adds 210 relative features, giving 431 dimensions.

The ensemble equally averages corresponding outputs of a raw-only
predictor and a raw-plus-relative predictor. It keeps an unaugmented
reference alongside the augmented comparison. This is the design rationale,
not a claim of statistically established superiority over relative-only
routing; the reported component comparison nearly matches the full model.

## Policy grid

For each candidate, the policy score is
`regression_weight * transformed_gain + p_positive - negative_penalty * p_negative`.
The frozen grid is the Cartesian product below, with the last varying
dimension changing fastest:

| Parameter | Values in enumeration order |
| --- | --- |
| Regression weight | `0.0, 0.5, 1.0, 2.0` |
| Negative-probability penalty | `0.5, 1.0, 2.0` |
| Score threshold | `-0.25, 0.0, 0.2` |
| Winner/runner-up margin threshold | `0.02, 0.05` |
| Maximum negative probability | `0.4, 0.5` |
| Minimum positive probability | `0.2` |

There are `4 * 3 * 3 * 2 * 2 * 1 = 144` policies. A route must remove at
least 40% of chunks to enter score ranking. Activation requires a score and
winner margin strictly above their thresholds, negative probability at most
its ceiling, and positive probability at least its floor. A sole eligible
route has an infinite winner margin. With no accepted route, the compressed
anchor is used. "Active" below means this learned route was accepted, not
that its eventual answer was correct.

## Development feasibility and risk

Compute the following indicators separately on complete LOTO and LOFO
out-of-fold predictions, using the final selected route or fallback for
each query. All thresholds below are development-selection settings, not
guarantees for unseen tasks.

**Feasibility indicator** is one only if all conditions hold:

- Every query has positive pool compression.
- Query-weighted mean compression is at least 55%.
- Every task's mean compression is at least 55%.
- The unweighted mean of task activation fractions is at most 70%.
- No task activates learned routing for more than 90% of its queries.

**Risk-feasibility indicator** is one only if all conditions hold:

- The worst task's mean F1 difference against Dense is at least -1.50
  percentage points (equivalently -0.015 on the 0-to-1 scale).
- Among active queries, the fraction with a negative F1 difference is at
  most 20%.
- Among active queries, the fraction with a positive F1 difference is at
  least 12%.

These indicators lead a lexicographic ordering; they are not undocumented
continuous penalty terms. In particular, ranking infeasible policies is
still defined. The frozen selected policy satisfies both indicators under
both validation schemes, as recorded in the model manifest.

## Exact objective order

For each validation scheme, maximize this tuple lexicographically:

| Position | Quantity (larger is preferred) |
| --- | --- |
| 1 | Feasibility indicator |
| 2 | Risk-feasibility indicator |
| 3 | Fraction of families with positive mean F1 difference |
| 4 | Minimum family mean F1 difference |
| 5 | Fraction of tasks with positive mean F1 difference |
| 6 | Minimum task mean F1 difference |
| 7 | Negative number of tasks with negative mean F1 difference |
| 8 | Positive-difference fraction among active queries |
| 9 | Negative of the negative-difference fraction among active queries |
| 10 | Mean query F1 difference after clipping each difference to [-0.05, 0.05] |
| 11 | Task-macro mean F1 difference |
| 12 | Query-weighted mean F1 difference |
| 13 | Query-weighted mean compression |

Development family means pool the queries belonging to a family; they are
not unweighted averages of its task means. Task means pool queries within a
task; task-macro means weight tasks equally. F1-difference entries in the
stored objective use percentage points. Query-level active signs use
`EPS=1e-12` on the 0-to-1 difference; family/task win and loss counts compare
their percentage-point means against the same numerical epsilon. If there
are no active queries, both active fractions are zero.

Take the componentwise minimum of the LOTO and LOFO tuples, followed by
their componentwise mean, giving a 26-component tuple. Maximize that tuple
lexicographically over the 144 policies. This does **not** mean choosing one
whole validation scheme as the worse one. If two policy tuples are exactly
equal, retain the first policy in the fixed grid order. The recorded winner
is zero-based index **116**:

```json
{
  "regression_weight": 2.0,
  "negative_penalty": 0.5,
  "score_threshold": 0.2,
  "route_margin": 0.02,
  "maximum_negative_probability": 0.4,
  "minimum_positive_probability": 0.2
}
```

The selector is then fitted on all 5,763 development queries and frozen
before formal evaluation. The same frozen route decisions are reused for
the tested Llama generator without target-model labels or retraining.
The [data inventory](development_inventory.md) identifies the development
tasks and separate evaluation subsets. The [training implementation](selector_training.md)
now supports final fitting and LOTO/LOFO final-policy selection from prepared
development inputs. It does not rerun the earlier nested development study
or the generator-level benchmark evaluation.

## 中文说明

保留池是每条路由确定最终证据之前的候选工作集，不是声称已经实现的跨查询
长期记忆。开发阶段对 144 种固定策略按可行性、风险、family/task 指标等
顺序进行字典序比较：先比较 LOTO/LOFO 各指标较小值组成的元组，再比较
各指标均值组成的元组，完全并列时保留固定枚举顺序中的第一项。所有阈值、
顺序和最终配置均已列出。新增训练入口可在用户提供的开发输入上训练新模型，
原论文冻结模型与策略不变，也没有根据正式评测重新调参。
