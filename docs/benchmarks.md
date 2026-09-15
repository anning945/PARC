# Benchmark Sources and Provenance

PARC does not redistribute benchmark data. Users should obtain each dataset
from its upstream project, respect the upstream terms, and keep the pinned
revision and file hashes recorded below when reproducing the paper setup.

## Four benchmark families

| Family | Upstream source | Pinned code/data reference | Official evaluation reference |
| --- | --- | --- | --- |
| BAMBOO | <https://github.com/RUCAIBox/BAMBOO> | code commit `f230f206148396c12774ef3a632df291d96dc6ab` | `evaluate.py`, binary F1 |
| ETHIC | <https://github.com/dmis-lab/ETHIC> | code commit `b26c399e494f9a44c97115a2b37503eeba87f1af`; Hugging Face revision `34d9a5c6ffd9b06b1eca75d2e7a437bc96d628c4` | `utils.py`, sample-level set F1 and official mean |
| HELMET | <https://github.com/princeton-nlp/HELMET> | code commit `af609c4d51b97fc35012099380aa889da961c42d`; original v1 data revision `c5b85e7f0d954ffe71b3fe5b6d1da17a11094b1b` | official HELMET evaluator; QA F1 is the primary quality metric |
| LongTableBench | <https://github.com/liyaooi/LongTableBench> | code commit `8db45ac102632cba3bcc1393e1023f0689da7d1c` | `eval/result_process.py`, structured-answer macro F1 |

These four families are the complete benchmark-family inventory used by the
paper-side release scope. The experiment reports the full eligible release
under each upstream task definition; it must not be replaced by a convenient
subset of successful samples.

## Licensing notes

- LongTableBench documents its dataset terms in the upstream README (CC BY
  4.0 was recorded for the pinned release).
- BAMBOO's pinned README claims MIT, but the pinned tree did not contain a
  separate license file; users should verify the upstream repository before
  redistribution.
- ETHIC's Hugging Face metadata records CC BY 4.0 for the dataset; the pinned
  code tree did not contain a separate code license file.
- HELMET's code and data terms follow the upstream repository and its original
  v1 release. The later v2 archive is not the paper's pinned data revision.

The PARC software license is independent from the licenses of these datasets.
The PARC repository must include a separate software `LICENSE` file before
public release.

## Reproduction boundary

This release records source provenance and the retrieval-only PARC routing
interface. It intentionally does not include benchmark questions, answers,
labels, raw contexts, model checkpoints, generated predictions, or per-example
scores. Complete end-to-end reproduction additionally requires upstream data,
the benchmark adapters, retriever/reranker configuration, generator setup,
and the official evaluators.

## 中文说明

PARC 不在仓库中重新分发 benchmark 原始数据。复现实验时应从上游项目
下载数据，遵守数据许可证，并使用表中固定的代码提交、数据版本和评测脚本。
四个 benchmark family 为 BAMBOO、ETHIC、HELMET 和 LongTableBench。
