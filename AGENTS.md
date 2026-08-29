# WAM Repository Guide

Repository-level conventions, and the default working instructions for AI agents. Goal: fast
research iteration, reproducible experiments, and code an outsider can read and run within an hour.
Full rationale and examples live in [`docs/CODE_STYLE.md`](docs/CODE_STYLE.md); this file is the
executable summary.

## Principles

1. Readability over generality: do not add abstraction for imagined requirements.
2. Explicit over implicit: config must be declared, errors must surface, tensor shapes must be written down.
3. Deleting is a contribution: remove unused branches, dead config, and "might need it later" interfaces.
4. These rules govern how we write, not what we research. When a rule blocks an experiment, run the experiment first, then fix the rule with the concrete case.

## Architecture boundaries

| Directory | Responsibility |
| --- | --- |
| `wam/` | Model families; must not depend on a specific benchmark, data path, or format |
| `backbone/` | Video backbone adapters; no action or benchmark logic |
| `modules/` | Reusable blocks; no special-casing of a specific backbone |
| `data/` | Dataset reading, normalization, caching |
| `training/` | Trainer, losses, flow utilities, training entry points |
| `eval/` | Benchmark-agnostic policy wrappers and generic inference |
| `examples/<bench>/` | Only env setup/teardown, observation conversion, action conversion, recipes, launch scripts |
| `vendor/` | Third-party code; every source needs a `SOURCE.md` with URL, commit, and local modifications |

- A new capability belongs to exactly one extension axis: model family, backbone, data source, or benchmark. Cross-axis changes must be justified in the PR.
- A benchmark adapter must not implement checkpoint loading, normalization, or generic inference. Past 300 lines, check whether responsibilities are misplaced.
- One file, one topic; a new topic gets a new file.
- Extract an abstraction only once a second real implementation exists; three implementations require a shared interface.

## Interfaces and data contract

- `WAMModel` implements `training_step`, `infer_action`, `infer_joint`.
- Every `BaseBackbone` abstract method states its full behavioural contract in the docstring, including what the returned object must provide.
- Models and the data layer communicate only through the sample dict, defined in `data/__init__.py`:
  - `video [B,3,T,H,W]` normalized to `[-1,1]`
  - `action [B,K,A]` normalized per the recipe
  - `context [B,L,D_text]`
  - Optional: `context_mask [B,L]`, `proprio [B,P]`, `action_is_pad [B,K]`
- Reading optional sample keys with `.get()` is part of the contract. Reading config with `getattr` is not.

## Configuration

- dataclass + YAML recipes. The dataclass is the single source of truth for fields, types, and defaults.
- YAML loading rejects unknown keys. Never `getattr(cfg, "x", default)` — use `cfg.x`.
- Give each field a one-line comment above the declaration, unless the name and type already say everything. Always comment implicit units, implicit scope (per-device vs global), and any special meaning of `None`/`0`/`-1`. Never restate the field name.
- Recipes contain no personal absolute paths; local paths come from CLI overrides or uncommitted scripts.
- Anything derivable from existing fields or from the checkpoint is not a new required field.
- All validation runs at the top of `main()` in `train.py` / `eval.py`; shell scripts carry no correctness checks.
- Fail hard on anything affecting correctness or reproducibility; warn on efficiency or reference-alignment issues.
- Shared defaults must run on a single machine; cluster, upload, and monitoring features default to off.
- Do not introduce Hydra until config composition and sweeps are a real maintenance burden.

## Errors and logging

- User input errors: `ValueError`. Internal invariants: `assert`. Unimplemented branches: `NotImplementedError`.
- Error messages state the field, the actual value, and what is allowed or how to fix it.
- Never continue after catching `Exception`. Only explicit optional-dependency imports, concurrency races, and telemetry isolated from computation may swallow an error.
- Document tensor shapes. Assert only high-risk invariants whose violation could be accepted silently,
  such as manual concat/split, attention-mask construction, and bf16/fp32 seams. Do not repeat an
  established contract at every consumer. Heavyweight per-step checks go behind a debug switch.
- Core package uses `logging`, never `print`. Operations over 5 seconds log both start and finish, with `%s` lazy formatting.
- Only rank 0 emits `INFO` under distributed training.
- The training log line carries step, per-term losses, lr, step time, throughput, and ETA. Log a full config summary at startup.

## Comments and code shape

- Comments explain why, why not, and when it breaks — never restate the code.
- Module docstrings for model families and backbones list the key features plus the paper and reference implementation.
- Tensor functions annotate input and output shapes using the shared alphabet `B,T,T',K,A,L,S,D`.
- Type annotations are required on public functions, dataclasses, and cross-module interfaces; short internal helpers are exempt.
- Naming: `snake_case` for functions and modules, `PascalCase` for classes, `UPPER_SNAKE` for constants, affirmative booleans.
- Discussion thresholds: functions 100 lines (hard 180), files 600 lines (hard 1200), width 120. Orchestration functions are exempt with an explanation; vendored code does not count.

## Experiments and Git

- Official recipes record at the top: training setup, schedule alignment, status, checkpoint, evaluation results with trial counts. Update or delete stale results when hyperparameters change.
- Comparisons pin explicitly: backbone and data versions, global batch, total samples, normalization statistics, training/data/eval seeds, inference steps, trials.
- The same checkpoint + recipe + seed must evaluate identically across runs.
- Commit subjects are lowercase imperative, no `feat:`/`fix:` prefix, at most 72 characters. Changes over 20 lines need a body covering why, blast radius, and verification.
- Squash the branch into one commit before opening an MR (`git reset --soft $(git merge-base HEAD origin/main)`), and push with `--force-with-lease`. Two commits are acceptable only for logically independent changes that each pass `runs/smoke.sh`.
- Personal branches `feature/dev_<initials>`, feature branches `feature/<topic>`. `main` always runs the smallest experiment.
- Data, weights, outputs, and personal scripts are never committed — use `.git/info/exclude`. `.gitignore` holds only team-wide rules.

## Testing and dependencies

- Test only: the data contract, numerical equivalence, determinism, and pure functions such as normalization and mask construction.
- `runs/smoke.sh` runs 20 training steps plus one evaluation on synthetic data in under five minutes, covering every model family. Run it before merging.
- GPU tests use `pytest.mark.skipif` and must not block developers without a GPU.
- A new runtime dependency needs a PR explaining why implementing it ourselves is unsuitable; dev tooling goes in the dev group.
- Pin minor versions of core training dependencies; commit the lock file.

## Definition of done

Before submitting, confirm:

- The change sits in the right layer, with no cross-axis coupling and no duplicated generic logic.
- Config fields are declared and commented; no unknown keys, absolute paths, or `getattr` defaults.
- No swallowed errors; long operations log; tensor shapes are documented and high-risk silent invariants are asserted.
- Recipes, results comments, and academic attribution track the behaviour change.
- Relevant tests and `runs/smoke.sh` pass.
- The branch is squashed to one commit.
- No dead code left that could have been removed along the way.

## Maintenance

- Change this file through a PR with a concrete case; substantive changes need at least one reviewer.
- Read it end to end on rotation the first Monday of each month; new members raise unclear points after two weeks; full review before open-sourcing.
- Change it immediately when a rule is repeatedly violated, blocks research, or its cited evidence has gone stale.
- A rule not cited in review for three months should be trimmed, rewritten, or deleted.
- Record rule changes at the end of this file: `date — what changed — why`.

### Change log

- 2026-08-28 — Distilled repository-level rules from `docs/CODE_STYLE.md` — reduce human maintenance and AI context cost.
- 2026-08-28 — Made shape assertions selective — avoid repeated contract checks that reduce readability.
