"""roger_server — the federated aggregation server.

Wire-compatible with the Roger gradient-sharing client (`roger.federated.transport` in the main
`roger` repo): it seals secure-aggregation cohorts, sums the masked uploads of the epoch's trainable
LoRA factor (masks cancel), folds η·mean(Δ) into a per-model cumulative global adapter, and broadcasts
it. Run it with `python -m roger_server`. The secure-aggregation + factor wire contract this shares
with the client lives in `secure_agg.py` + `delta.py` here (kept in lockstep with the client's copies
by hand).
"""
from roger_server.aggregate import Aggregator
from roger_server.app import create_app

__all__ = ["Aggregator", "create_app"]
