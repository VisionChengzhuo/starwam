# WAM Codebase Style Guide

Coding conventions for the world-model research codebase evolved from StarWAM. The goal is a
**lightweight research codebase that is easy to extend, easy to develop in, and easy to read** —
by the time we open-source, an outside reader should be able to understand the overall structure
and run their first experiment within an hour.

> Starting point: StarWAM · Style references: nanochat, openpi · Explicitly **not** adopting
> production-grade defensive engineering · Version 1.8 (2026-08-28)
>
> This file is the canonical copy. A rendered Chinese version is also available as a web page:
> <https://claude.ai/code/artifact/bb8135da-1149-4714-9ea0-7e785ec95d14> — same rules, kept for
> reading and sharing. When the two disagree, this file wins.

---

## Contents

- [§0 Guiding principles](#0-guiding-principles)
- [§1 Architecture and layout](#1-architecture-and-layout)
- [§2 Abstraction and extension](#2-abstraction-and-extension)
- [§3 Configuration](#3-configuration)
- [§4 Error handling](#4-error-handling)
- [§5 Logging](#5-logging)
- [§6 Comments and documentation](#6-comments-and-documentation)
- [§7 Naming and shape](#7-naming-and-shape)
- [§8 Reproducibility](#8-reproducibility)
- [§9 Git](#9-git)
- [§10 Testing](#10-testing)
- [§11 Dependencies](#11-dependencies)
- [§12 Review checklist](#12-review-checklist)
- [§13 Maintaining this document](#13-maintaining-this-document)

---

## §0 Guiding principles

When this document does not cover your situation, decide using these three. They outrank every
specific rule below.

**P1 — Readability over generality.** The first reader is you in three months, and the colleague
who just joined. An abstraction added to cover an imagined third case usually costs more than it
returns.

**P2 — Explicit over implicit.** Config fields must be declared, errors must be raised, shapes must
be written down. Do not use fallback defaults to paper over a caller's mistake.

**P3 — Deleting code is an achievement.** A negative-diff PR deserves as much credit as a feature
PR. Branches no experiment uses, config options nobody sets, interfaces kept "in case we need them"
— delete them.

> **What this document governs.** It constrains *how you write*, not *what you think*. Model design,
> loss formulation, hyperparameter choices are all out of scope. Wherever this document gets in the
> way of the research itself, run the experiment first and open an issue to change the document
> afterwards.

---

## §1 Architecture and layout

StarWAM's three-layer split is the most valuable asset in this codebase; this section freezes it
into rules. The core principle: **the core package knows about no specific benchmark, and a
benchmark adapter implements no general-purpose logic.**

### 1.1 Directory responsibilities

| Path | Holds | Never holds |
| --- | --- | --- |
| `wam/` | Model families (MoT / Shared-DiT / feature-conditioned) | Data formats, benchmark names, file paths |
| `backbone/` | Video backbone adapters (Wan2.2 / Cosmos3 / …) | Anything action-related |
| `modules/` | Reusable building blocks: DiT block, MoT, schedulers | Special cases for a particular backbone |
| `data/` | Dataset reading, normalization, text caching | Model-structure code |
| `training/` | Trainer, losses, flow utilities, entry points | Benchmark-specific evaluation |
| `eval/` | Benchmark-agnostic closed-loop policy wrappers | Simulator imports |
| `vendor/` | Third-party implementations copied from upstream, each directory labelled with source and commit | Any code we wrote ourselves |
| `examples/<bench>/` | Everything unique to that benchmark: env interaction, observation conversion, recipes, launch scripts | Checkpoint loading, normalization, the generic inference loop |

### 1.2 A benchmark adapter contains only what is specific to that benchmark

The test: **would this code have to change if we switched benchmarks?** If not, it belongs in the
core package.

> **Lesson from StarWAM.** `examples/libero/rollout.py` and `starwam/eval/policy.py` contain three
> fully duplicated functions (`_strip_known_prefixes`, `_extract_checkpoint_state`,
> `_denormalize_action`). None of them has anything to do with LIBERO; there should be one copy.
> Add three more benchmarks and it becomes five copies.

A new benchmark adapter should contain only three kinds of code: **environment setup/teardown**,
**observation → model input**, and **action → environment**. Everything else calls into
`eval/policy.py`. If your adapter exceeds 300 lines, something is in the wrong place.

### 1.3 Vendored code must be isolated

`backbone/wan22.py` is 2246 lines, of which roughly 1670 are the T5 encoder and 3D VAE copied from
the official Wan repository and only 576 are our own adapter layer. Mixing them means noisy greps,
no way to diff against upstream, and newcomers who cannot tell which half they are allowed to edit.

Rule: third-party implementations go in `vendor/<upstream>/`, with a `SOURCE.md` in each directory
recording the repository URL, the commit hash, and the list of modifications we made. The adapter
layer stays in `backbone/` and may only import — never copy.

### 1.4 One file, one topic

New capabilities get a new file rather than being appended to an existing one. Adding DROID support
means `data/droid.py`, not 300 more lines and an `if self._is_droid` branch inside
`data/lerobot.py`.

---

## §2 Abstraction and extension

This codebase exists to compare different WAM architectures under one training pipeline, so where
the abstraction boundaries sit determines how believable those comparisons are. But abstraction has
a cost — this section covers both where abstraction is required and where it is forbidden.

### 2.1 Four orthogonal extension axes

Every new capability must land on exactly one of these axes. Coupling across axes is a design
accident.

| Axis | How to extend | Must not affect |
| --- | --- | --- |
| Model family | Subclass `WAMModel`, implement `training_step` / `infer_action` / `infer_joint` | Any backbone code |
| Video backbone | Subclass `BaseBackbone`, implement the 5 abstract methods | Any model-family code |
| Data source | Implement the shared sample-dict contract (see §2.3) | Trainer and model families |
| Benchmark | Write an adapter under `examples/` | Any file in the core package |

Adding Cosmos3 should touch only `backbone/`; adding DROID should touch only `data/` plus one new
recipe. If you find yourself having to change two axes at once, explain why in the PR first.

### 2.2 Write contracts into the docstring, and write them in full

`BaseBackbone.get_dit()` is the model to follow — it states not just what is returned, but what the
returned object must satisfy:

```python
"""Return the DiT (video expert) module for MoT composition.

The returned module must have:
- .blocks: nn.ModuleList of DiTBlock-compatible blocks
- Each block must have .get_qkv() and .post_attention() methods
"""
```

Do the same for every new abstract method. Keep and propagate the note at the top of
`modules/mot.py` — "a backbone should adapt to MoT's contract rather than MoT being modified" — it
is what stops later contributors from pushing special cases into the generic layer.

### 2.3 The data contract: sample-dict keys are a global agreement

Model families and data sources communicate through a single dict. Its key set is a **cross-module
public contract**, defined in the module docstring of `data/__init__.py`. New keys must be called
out in the PR description.

```python
# Contract maintained in data/__init__.py (illustrative)
# Required
#   video        [B, 3, T, H, W]   normalized to [-1, 1]
#   action       [B, K, A]         K = chunk_size, normalized per the recipe
#   context      [B, L, D_text]    T5 text embeddings (from cache)
# Optional
#   context_mask [B, L]            absent means all-ones
#   proprio      [B, P]            proprioceptive state, normalized
#   action_is_pad[B, K]            True marks a padding step
```

A model family reading optional keys with `sample.get(...)` is fine — that is part of the contract.
Reading **config** with `getattr` is forbidden (§3.3). Do not conflate the two.

### 2.4 Do not add an abstraction layer for a single experiment

This is nanochat's most important lesson. The test: **extract an abstraction if and only if a second
implementation already exists.** One implementation is one class; two implementations start the
conversation about a base class; three implementations require one.

```python
# ✗ Don't
class BaseNoiseSchedule(ABC):
    @abstractmethod
    def sample(self): ...

class FlowMatchSchedule(BaseNoiseSchedule):
    ...  # the only implementation today
```

An abstraction reserved for an imagined second schedule forces every reader to hop through one extra
layer to reach the real logic.

```python
# ✓ Do
class FlowMatchScheduler:
    """Flow matching noise schedule with logit-normal timestep sampling."""
    ...  # just write it
```

When the second one actually shows up, extract the base class then — by that point you know what is
genuinely shared, and the abstraction will be a better fit.

---

## §3 Configuration

We keep StarWAM's dataclass + YAML recipe design (and explicitly do not introduce Hydra), but fix
its biggest flaw: the **single source of truth** for config fields being diluted by `getattr`
fallback defaults.

### 3.1 The dataclass is the single source of truth

Every config field must be declared in a dataclass in `config.py`, with a type annotation, a default
value, and a one-line comment (format in §3.2). Keep the behaviour of rejecting unknown keys when
loading YAML — it surfaces typos in the first second.

### 3.2 Give config fields a one-line comment, unless they are self-explanatory

**Config fields are the only user interface this codebase has**: nobody is going to read
`trainer.py`, but everyone has to read `config.py` to know what to fill in. openpi's `TrainConfig`
has a one-line comment above nearly all of its thirty fields, and that is the main reason its
989-line `config.py` is still readable. **Write one by default; skip it when the field speaks for
itself.**

The comment goes on the **line above** the declaration (a trailing comment cannot fit a full
sentence). It should answer at least one of the following; **if it answers none of them, the field
does not need a comment**:

- **What the field is**, especially when the name is ambiguous (is `batch_size` per-device or
  global?)
- **Units or valid range** — steps or epochs, seconds or milliseconds, pixels or normalized
  coordinates
- **Side effects of turning it up or down**, and any special meaning of `None`

```python
@dataclass
class TrainingConfig:
    # Per-rank micro-batch. Global batch = batch_size × grad_accum × world_size.
    batch_size: int = 1
    # Raise this before lowering batch_size when you run out of memory.
    gradient_accumulation_steps: int = 8
    # full = all parameters / lora = adapters only / staged = progressive unfreezing.
    strategy: str = "full"
    # None means derive from num_epochs and dataset size; see §3.5.
    max_steps: int | None = None

    # The names below already say everything; do not pad them with comments:
    seed: int = 42
    output_dir: str = "outputs"
    resume: bool = False
```

The test is one sentence: **from the name and type alone, can you fill in the right value?** If yes,
skip the comment. A comment that restates the field name (`# batch size`) is worse than none — it
makes readers think documentation already exists.

Conversely, these three cases look self-explanatory but are not, and always need a comment however
short the name:

- **Implicit units** — is `warmup: int` steps or epochs? Is `timeout` seconds or milliseconds?
- **Implicit scope** — is `batch_size` per-device or global? Is `save_interval` in steps or epochs?
- **`None` / `0` / `-1` carrying special meaning**

The author makes the call. The review test: *if a reviewer has to ask "what does this field mean?"
before they can keep reading, it needed a comment.*

### 3.3 No `getattr(config, "x", default)`

> **Where we are.** StarWAM has 139 `getattr` calls repo-wide (`trainer.py` 25, `builder.py` 24).
> The consequence is that one field's default lives in the dataclass *and* in several call sites;
> change one and miss the others and you get a silent behavioural split.

```python
# ✗ Don't — the default is scattered across call sites, and no reader can answer
#            "what does this field actually default to?"
norm = getattr(cfg.data, "action_norm_mode", "minmax")
dim  = getattr(cfg.framework, "proprio_dim", None)

# ✓ Do — the field is guaranteed to exist by the dataclass, and the default
#         lives in exactly one place: the declaration.
norm = cfg.data.action_norm_mode
dim  = cfg.framework.proprio_dim
```

**The one exception**: reading optional attributes off *other* objects (models, backbones,
third-party return values) may use `getattr`. Reading config never does.

### 3.4 No absolute paths in recipes

With open-sourcing ahead, a `/home/<someone>/...` in a recipe is a hard defect. The convention:

- Recipes committed to the repository use the placeholder `/path/to/xxx`, or read an environment
  variable
- Real local paths are passed via `--override`, or live in a launch script that is **not committed**
- Personal paths, datasets, weights and output directories go in `.git/info/exclude`, not
  `.gitignore` — the latter is shared with everyone and should not encode anyone's machine layout

Add a CI or pre-commit check: reject a merge when the diff contains a string starting with `/home/`
or `/mnt/`.

### 3.5 Do not make users supply what can be derived

nanochat derives width, head count, learning rate and step count from a single `--depth`. We do not
need to go that far, but the direction is the same: **for every new required field, first ask
whether it can be computed from existing ones.**

StarWAM already has a good example in `BackboneInfo` — every dimension is inferred from the
checkpoint, with a comment that says `Never manually set by user`. Keep that convention for new
backbones.

### 3.6 Validation runs at the program entry point

> **Today's gap.** `validate_preset` is well written, but it is only called from four 8-GPU launch
> scripts; `training/train.py` never references it. Anyone running `python -m ...train` directly
> bypasses every check.

Rule: all config validation runs at the top of `main()` in `train.py` / `eval.py`. Shell scripts
carry no validation responsibility. Validation has two levels:

- **Hard failure** — anything affecting correctness or reproducibility (model dimensions,
  normalization mode, data contract)
- **Warning** — anything affecting efficiency or alignment with a reference implementation
  (`num_workers`, batch composition, epoch count)

### 3.7 New defaults target a single machine

Some of us develop on one GPU and others train on many nodes. A default written into a shared recipe
must be the one that **runs as-is on a single machine**. Switches that only make sense with cluster
resources (object-storage upload, remote monitoring, multi-node communication tuning) default to
off, and are turned on explicitly by cluster launch scripts.

The test case comes from a real incident: a change set the shared recipe's `save_total_limit` from 3
to `null` and simultaneously enabled object-storage upload by default, on the reasoning that
"uploading keeps a copy, so we don't need to delete locally." Single-GPU users had no upload target,
and no disk that could absorb 40 GB every 2000 steps. *That change has since been reverted from
main.*

---

## §4 Error handling

This is where this guide diverges most from industrial style. **We want to locate problems fast, not
to run unattended for long periods.** In research code, an exception swallowed by an `except` can
mean discovering two days later that an entire run's data was wrong.

### 4.1 `raise` faces the user, `assert` faces us

| Situation | Use | Why |
| --- | --- | --- |
| Input the user can get wrong: config fields, paths, CLI arguments | `raise ValueError` | Needs a readable path to a fix |
| Internal invariants: tensor shapes, divisibility, call ordering | `assert` | A failure means *our* bug; the stack matters more than the wording |
| Unimplemented branch | `raise NotImplementedError` | Clearly distinct from "something went wrong" |

The two reference codebases sit at opposite ends: StarWAM has 123 `raise ValueError` and 8 `assert`;
nanochat has 79 `assert` and 16 `raise`. We take the middle — **`raise` at config-facing boundaries
and use `assert` selectively for internal invariants that could otherwise fail silently** (§4.2).

### 4.2 Assert only where violations can fail silently

Shape comments document the contract (§6.2); runtime assertions protect selected invariants. Do not
mechanically pair the two. Repeating the same shape check at every consumer makes code longer and
obscures the computation without adding useful coverage.

```python
assert q.dtype == k.dtype == v.dtype == dtype
assert mask.shape == (batch_size, 1, seq_len, cache_size), mask.shape
assert k.shape[1] == 1, k.shape
```

Passing the variable itself as the second argument (`, mask.shape`) is enough — the traceback prints
it. **Do not craft prose for an assert**; that is `raise` territory.

Strong candidates for an assert are invariants whose violation could be accepted and silently change
the result:

1. **After manual concat / split** — check segment lengths when a wrong split would still produce
   valid tensors and silently train the wrong model.
2. **After constructing an attention mask** — check dtype and shape because broadcasting or mixing
   boolean and additive masks may otherwise look valid.
3. **At dtype boundaries** — check bf16 / fp32 seams, especially where vendored and in-house code
   meet (§1.3).
4. **At untrusted component boundaries** — check an external or vendored return value only when the
   downstream operation could accept the wrong shape or dtype.

Do not repeat an established sample-dict contract at every trainer and model consumer. Do not assert a
shape immediately before an operation that already raises a clear error for the same mismatch. A module
boundary alone is not a reason to add a check.

Do not assert what the type annotations already guarantee
(`assert isinstance(cfg.batch_size, int)`), and put heavyweight per-step checks behind a switch:

```python
if cfg.training.debug_checks:          # False by default
    assert torch.isfinite(loss), f"loss={loss.item()} at step {step}"
```

`assert` is stripped by `python -O`. We do not use `-O`, but by the same token, **a check that must
never be skipped has to be a `raise`**.

### 4.3 Three parts to an error message

The format: **which field + what you gave + what is allowed**. StarWAM's existing style is the
standard:

```python
raise ValueError(
    f"Unknown model_family={model_family!r}; allowed: {sorted(allowed)}"
)
raise ValueError("data.root or data.dataset_dirs must be set for dataset_type='lerobot'")
```

Note the `!r` — it leaves empty strings and stray whitespace nowhere to hide.

### 4.4 No defensive `try/except`

```python
# ✗ Don't
try:
    stats = load_action_stats(path)
except Exception as e:
    logger.warning(f"stats load failed: {e}")
    stats = None   # training continues with unnormalized actions
```

The run will finish, the loss will go down, the result will be wrong, and you will not find out
until the rollout success rate looks strange.

```python
# ✓ Do
stats = load_action_stats(path)   # let it blow up
```

The stack trace points straight at the problem. Five minutes instead of two days.

**Never continue execution after `except Exception`.** When you genuinely need to catch, catch the
specific exception type and `raise` a better-informed one from the block. Only three situations may
truly swallow an exception:

1. **Optional dependency imports** — `except ImportError`, print installation guidance, then `raise`
2. **Multi-process / file-lock contention** — `except FileExistsError` and similar explicit
   concurrency semantics
3. **Pure telemetry** — metric collection used only for printing, feeding into no computation, and
   isolated in its own function

Every other `try/except` is questioned by default in review, and the author has to explain why that
failure is safe to ignore.

---

## §5 Logging

StarWAM currently runs three regimes at once: the trainer uses `logger` (37 sites), `wan22.py` uses
`print` (15 sites), and `mot_wam.py` and `lerobot.py` emit nothing at all. Unify them.

### 5.1 Use `logging`, never `print`

```python
logger = logging.getLogger(__name__)   # fixed first line of every module
```

The only exception is body output from an interactive user-facing script (e.g. a chat CLI). A
`print` in the core package is sent back in review.

### 5.2 Anything taking over 5 seconds needs a start signal

> **Real case.** Loading Wan2.2-5B's DiT weights reads three safetensors shards in complete silence,
> printing a single `Loaded DiT weights ...` at the end. First-time users assume it has hung.

Model loading, dataset scanning, statistics computation, text-cache generation — anything that may
exceed 5 seconds gets a matched pair of log lines:

```python
logger.info("Loading DiT weights from %s ...", model_dir)
info = dit.load_pretrained(model_dir, dtype=dtype)
logger.info("Loaded DiT weights (num_loaded=%d, missing=%d)",
            info["num_loaded"], len(info["missing_keys"]))
```

Use `%s` lazy formatting rather than f-strings — a filtered-out log line should not pay formatting
cost.

### 5.3 Only rank 0 speaks

Set non-zero ranks to `WARNING`. Debug output that genuinely needs every rank carries an explicit
rank prefix and is off by default.

### 5.4 Fixed fields in the training log

One line every `log_every`, fields in a fixed order, so it greps cleanly and compares by eye across
runs:

```
step 1200/21370 | loss 0.3841 (video 0.3102 action 0.0739) | lr 9.42e-05 | 1.21s/step | 6.6 samples/s | eta 6.9h
```

Print a config summary once at startup (dataset size, effective global batch, derived `max_steps`
and warmup) — it is the first place you look when reconstructing how a run was configured.

---

## §6 Comments and documentation

Both reference codebases agree here: comments explain **why**, they do not restate what the code
does. Shape comments are a hard requirement in this kind of code, not an option.

### 6.1 A module docstring is a feature list

nanochat's `gpt.py` opens with 11 architectural choices; the file header alone tells you what the
model is. Do the same for new model families and backbones:

```python
"""MoT WAM: per-layer joint attention between a video expert and an action expert.

Notable features:
- Separate video / action experts, per-layer mixed Q/K/V
- First-frame clean pinning (the first latent frame is left un-noised)
- Configurable first_frame / full_video action conditioning
- Velocity target noise - sample, weighted flow matching
References: Fast-WAM (arXiv:xxxx.xxxxx), Motus
"""
```

### 6.2 Shape comments are mandatory

Every function that produces or transforms tensors annotates the shapes of its arguments and return
values. Use one shared alphabet for named dimensions, declared at the top of `modules/__init__.py`:
`B` batch, `T` video frames, `T'` latent steps, `K` action chunk, `A` action dim, `L` text length,
`S` token sequence, `D` hidden dim.

```python
def encode_video(self, video: Tensor) -> Tensor:
    """video: [B, 3, T, H, W] in [-1, 1] -> latents: [B, C, T', H', W']"""
```

Do not mechanically pair every shape comment with an assert. Add a runtime check only when the
wrong shape could be accepted and silently change the computation (§4.2).

### 6.3 Comment the counter-intuitive parts

Do not write `# compute the loss`. Write one of three things: **why this way**, **why not that
way**, **when it breaks**.

```python
# We rotate by -theta, the transpose of the textbook convention. Functionally
# equivalent (only the relative q/k rotation matters); kept for checkpoint compat.

# The first latent frame is left un-noised: the action expert depends on a clean
# first frame as its observation condition.
```

### 6.4 Section long files with `# ---`

Files over 200 lines are split by topic, with one consistent separator:

```python
# -----------------------------------------------------------------------------
# Action normalization
```

### 6.5 Cite the source

Every model family's docstring names the paper and the reference implementation it follows. This is
both academic honesty and the basis for judging in review whether an implementation is faithful to
the original method.

---

## §7 Naming and shape

### 7.1 Naming

- Modules and functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`
- Private helpers start with `_`; module-level visibility is enough, no `__all__` bookkeeping
- Config field names are spelled out, never abbreviated: `action_norm_mode`, not `act_nm`
- Tensor variables may be short (`q`, `k`, `v`, `x`) — the shape comment already carries the meaning
- Boolean fields are affirmative: `normalize_actions`, not `disable_action_norm`

### 7.2 Type annotations at the boundaries

Public functions, config dataclasses and cross-module interfaces must be annotated. Short helpers
inside a module are not required to be. Use modern syntax — `int | None`, `list[str]`,
`dict[str, Tensor]` — with `from __future__ import annotations` at the top of the file.

We are not chasing 100% coverage: nanochat has only 32 annotations repo-wide and still reads
clearly, because the shape comments carry most of the information.

### 7.3 Size limits

These are not gates, they are **thresholds that trigger a conversation**. Past the soft limit a
reviewer may ask "should this be split?"; past the hard limit the author explains in the PR why it
cannot be — a good explanation is enough to merge.

| Subject | Soft limit | Hard limit | Notes |
| --- | --- | --- | --- |
| Function | 100 lines | 180 lines | Naturally linear orchestration (training loop, `main()`) is exempt |
| File | 600 lines | 1200 lines | Vendored code does not count |
| Line width | 120 chars | — | Not enforced by wrapping, but do not write 200-char lines |

> **Where we are (main, 2026-08-28).** Longest functions: `rollout.main()` 157 lines,
> `trainer._run_eval` 142. Largest files: `trainer.py` 905, `lerobot.py` 794, `rollout.py` 644
> (vendored `wan22.py` at 2246 excluded). Existing code is essentially all within the hard limits;
> **no refactoring for compliance is needed**. The one file that should actually change is
> `rollout.py` — it is long because it absorbed logic that belongs in `eval/` (§1.2), so the reason
> to split it is misplaced responsibility, not line count.

Orchestration functions (the training loop, `main()`) get more room — chopping them into ten small
functions reads worse. But timing, logging and evaluation calls inside the loop should be extracted
into methods so the loop body stays linear.

---

## §8 Reproducibility

The technical report needs "a completely fair systematic comparison." Reproducibility does not come
from good intentions; it comes from conventions.

### 8.1 A recipe is the experiment record

StarWAM's practice of writing results into a comment block at the top of the recipe is kept and
formalized. Every official recipe carries:

```yaml
# ============================================================
# Setup:      backbone / model family / dataset and scale / normalization
# Schedule:   batch, grad_acc, world_size relative to the reference implementation
# Training:   steps completed, checkpoint used for evaluation
# Results:    per-suite scores + summary, with trial count and inference steps
# ============================================================
```

An unfinished recipe says `STATUS: in progress`. "The comment says 97%, but that was three changes
ago" is not acceptable — when you change a hyperparameter, update the comment or delete the results
block.

### 8.2 Controlling variables in comparisons

When comparing architectures, the following are written explicitly in the recipe and listed in the
report. They may not be left to defaults that "happen to match":

- Backbone weight version, dataset list and version, effective global batch, total training samples
- Normalization mode and the statistics file (the same `action_stats.json` reused across
  experiments)
- Random seed, inference steps, number of evaluation trials

Consider adding `scripts/diff_recipes.py` to print a field-level diff between two recipes, and paste
its output when reviewing a comparison.

### 8.3 Seeds and determinism

Training, data splitting and evaluation each have their own seed field. The same checkpoint + the
same seed + the same recipe must produce the same result across two runs. Failing that is a bug, not
"normal variance."

---

## §9 Git

### 9.1 Commit messages

Subject line in the imperative, lowercase, at most 72 characters, with no `feat:` / `fix:` prefix —
we are a small team, and the classification value is lower than the subject width it consumes.

**A body is required for any change over 20 lines.** StarWAM has had 700-line changes with a
one-line subject; nanochat spells out the blast radius and how it was verified. Follow the latter:

```
fix token_bytes calculation to use raw token bytes

The previous round-trip through str corrupted tokens that are not valid
standalone UTF-8 (bytes >= 0x80 became the 3-byte U+FFFD). 193 of 32768
tokens were affected; new runs will report bpb about 0.05% higher than old ones.

Verified on d12: no other change to the loss curve.
```

The body answers: **why the change**, **how far it reaches**, **how it was verified**.

### 9.2 Squash to one commit before opening a merge request

`wip`, `fix typo` and `revert previous` add nothing to main's history and make `git log`,
`git bisect` and `git revert` all harder to use. **One change on main is one commit.**

Squash locally before opening the MR:

```bash
# Option 1: interactive rebase, also lets you reorder
git rebase -i origin/main

# Option 2: flatten the whole branch in one go (faster, recommended)
git reset --soft $(git merge-base HEAD origin/main)
git commit          # write subject and body per §9.1

# A branch that was already pushed needs to overwrite the remote
git push --force-with-lease
```

Use `--force-with-lease`, not `--force` — the former refuses when the remote has commits you have
not seen, which is what stops you from overwriting someone else's work. **Never squash a branch
someone else is using.**

After squashing, the commit body is the *only* record of the change, so §9.1's requirements on the
body genuinely matter here. Dead ends worth remembering go in the body; the rest disappears with the
squash — which is the point of squashing.

**Exception**: when one MR contains two logically independent changes (e.g. "refactor first, then
build on the refactor"), two commits are acceptable, but each must pass `runs/smoke.sh` on its own.
If the two changes have no dependency between them, the right answer is two MRs. If you forget to
squash you do not need to reopen the MR — GitLab's *Squash commits* option flattens on merge — but
squash locally by default, so you see the exact commit that will land on main before you submit it.

### 9.3 Branches

Personal development branches are `feature/dev_<initials>`; feature branches are
`feature/<topic>`. `main` stays able to run the smallest experiment at any time.

### 9.4 What not to commit

Datasets, weights, output directories and personal launch scripts go in `.git/info/exclude` (local,
uncommitted, applies across branches). `.gitignore` holds only rules that are true for everyone.

---

## §10 Testing

We do not write many unit tests — research code changes constantly, and testing volatile logic is a
liability. But four categories are nearly impossible to debug when wrong, and must be tested.

### 10.1 Test only these four

| Category | Example | Why it must be tested |
| --- | --- | --- |
| Data contract | sample-dict keys, shapes, normalization round-trip | Failure is silent; results just get worse |
| Numerical equivalence | Loss bit-identical before/after a refactor, CPU and GPU paths agreeing | The only safety net a refactor has |
| Reproducibility | Two rollouts with the same seed match | Directly underwrites the report's credibility |
| Pure-function boundaries | Normalization/denormalization, attention mask construction | Easy to write, easy to verify, almost never changes |

Model forward numerics, whether training converges, evaluation scores — none of these get tests.
They are covered by the smoke script and experiment logs.

### 10.2 The smoke script matters more than unit tests

Maintain `runs/smoke.sh`: synthetic data + smallest model + 20 training steps + one evaluation,
finishing in under five minutes, covering every model family. **It must pass before merging to
main** — it catches more than twenty unit tests would. Hardware-dependent tests are skipped with
`@pytest.mark.skipif` so that people without a GPU can still run the suite.

---

## §11 Dependencies

### 11.1 New dependencies get scrutiny

nanochat has 9 runtime dependencies and keeps deleting them. Our position: **a new runtime
dependency needs a PR explaining why a few dozen lines of our own code will not do.** Anything used
only during development (plotting, notebooks, formatting) goes in the dev group. The reason is not
fastidiousness — after open-sourcing, every dependency is an installation obstacle for users and
another supply-chain surface.

### 11.2 Version pinning

Core training dependencies (torch, acceleration libraries) are pinned to a minor version; utility
libraries use `>=`. The lock file is committed.

---

## §12 Review checklist

Go through it item by item. Cite the rule number when raising a problem ("this violates §3.3") —
building that habit is also what gives §13.1's survival check something to measure.

- [ ] New code sits in the right directory layer; the core package does not know about a specific benchmark; vendored code is not in the same file as ours (§1.1–§1.3)
- [ ] No abstract base class introduced for a single implementation (§2.4)
- [ ] New config fields are declared in the dataclass, and read without a `getattr` fallback (§3.1, §3.3)
- [ ] New config fields carry an explanatory comment, or are genuinely self-explanatory; no comments that merely restate the field name (§3.2)
- [ ] No absolute paths in the diff (§3.4)
- [ ] New switches default to something a single machine can run (§3.7)
- [ ] No new `try/except` that swallows an error (§4.4)
- [ ] No new `print`; long operations log both start and finish (§5.1, §5.2)
- [ ] Tensor shapes are documented; asserts cover only high-risk invariants that could fail silently (§6.2, §4.2)
- [ ] No new function past the soft size limit (§7.3)
- [ ] Recipes whose hyperparameters changed have their results comment updated or removed (§8.1)
- [ ] The commit body says why, how far it reaches, and how it was verified (§9.1)
- [ ] The branch has been squashed to one commit (§9.2)
- [ ] `runs/smoke.sh` passes (§10.2)
- [ ] Anything dead worth deleting along the way — unused code, "might need it later" interfaces (P3)

---

## §13 Maintaining this document

This document was written when the codebase had three model families and one benchmark. After DROID,
Cosmos3 and more simulation environments land, some rules will stop being true. **Maintain it like
code; do not enshrine it.**

### 13.1 When to change it

No "review it in the weekly meeting" ritual — that gets skipped. Tie it to things that will happen
anyway:

| When | What | Cost |
| --- | --- | --- |
| First Monday of the month | One person on rotation reads it end to end and opens an improvement PR (which may be "no change needed") | 20 min |
| Two weeks after someone joins | The new person raises everything they read but did not understand — they are the only ones who can find the blind spots | One conversation |
| Before the open-source release | Check: do rules assume our internal environment, do the examples still run, are there internal paths or names | Half a day |
| After the report is finalized | Fold the reproducibility problems the comparison exposed back into §8 | 1 hour |

The rotation matters more than its conclusions — it guarantees everyone has read the whole thing at
least once. While on rotation, apply P3 to the document itself and ask: **which rules were cited in
review over the past month?** A rule not cited for three months is either fully internalized (trim
it to one line) or ignored by everyone (delete or rewrite it). A longer document is not an
achievement; rules that actually get applied are.

Do not wait for the rotation. Open a PR immediately on any of these:

- **The same rule keeps getting violated** — cited three or more times and people still trip on it
  means the problem is likely the rule: either it is unclear, or it is fighting a real need
- **An argument the document does not cover** — two people doing the same thing differently and both
  making sense means a rule is missing. Settle it in the PR, then write the conclusion down
- **A new extension axis or class of infrastructure** — add the rule that fixes where its interface
  lives before writing the implementation; §2.3's data contract usually has to grow too
- **A rule is obstructing the research** — change the document, do not make the research take a
  detour. Readability outranks generality, but neither outranks getting the experiment done
- **Cited evidence has gone stale** — line counts, `getattr` counts and similar statistics in here
  are a snapshot of one moment; update them once fixed, so they do not become archaeology
- **An external contributor asks a question** — after open-sourcing, every "why is it written this
  way?" marks a place the document was not clear

### 13.2 How to change it

- The document lives in the repository as `docs/CODE_STYLE.md` and follows the same PR process as
  code
- Changing a rule needs at least one reviewer; wording, examples and updated statistics can be
  merged directly
- Every substantive change adds a row to the change log below, saying **what changed** and **why**
- Only ever growing makes the document useless — deleting a rule deserves as much credit as adding
  one

### 13.3 Change log

| Version | Date | Change |
| --- | --- | --- |
| v1 | 2026-08-28 | First version. Based on StarWAM as it stands and nanochat's simplicity practices, explicitly not adopting production-grade defensive style. |
| v1.1 | 2026-08-28 | Size limits relaxed and reframed as "thresholds that trigger a conversation"; code statistics updated against the rolled-back main; sequence-packing rules removed (not being implemented near-term). |
| v1.2 | 2026-08-28 | Removed the original §8.2 (`dev/LOG.md` experiment log). Experiment records are carried by wandb and the recipe header comment for now. |
| v1.3 | 2026-08-28 | Added §9.2 "squash to one commit before opening a merge request"; checklist updated. |
| v1.4 | 2026-08-28 | After reading openpi, added §3.2 (one-line comments on config fields) and §4.2 (pin shapes and dtypes with `assert`); subsequent rules in §3/§4 renumbered. |
| v1.5 | 2026-08-28 | §3.2 relaxed from "every field" to "by default, self-explanatory ones excepted", with the decision test and the three must-write cases. |
| v1.6 | 2026-08-28 | Trimmed roughly 10%: condensed the examples in §3.2/§4.2, tightened the evidence notes; "the three permitted uses of try/except" folded into §4.4 and "rule survival check" into §13.1; deleted the anti-pattern quick-reference section (redundant with §12). Rules and criteria unchanged. |
| v1.7 | 2026-08-28 | Translated to English and exported to `docs/CODE_STYLE.md` as the in-repo canonical copy; linked the rendered Chinese web version from the header. |
| v1.8 | 2026-08-28 | Made shape and dtype asserts selective instead of mandatory at every boundary; repeated contract checks made code longer without improving readability. |

*Add a row here on your next change.*

---

This guide governs how we write, not the research itself. When a rule is contested, discuss the
concrete case in a PR and change the document once you agree — the document follows practice, not
the other way round. Maintenance is covered in §13; the code statistics quoted throughout are a
snapshot from 2026-08-28, so update them when they go stale.
