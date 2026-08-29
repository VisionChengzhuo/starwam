"""StarWAM evaluation utilities (benchmark-neutral).

See :mod:`starwam.eval.policy` for the shared closed-loop policy wrapper.
LIBERO keeps its own ``examples/libero/rollout.py``; new benchmarks build a
thin adapter on top of :class:`starwam.eval.policy.StarwamPolicy`.
"""

from starwam.eval.policy import StarwamPolicy

__all__ = ["StarwamPolicy"]
