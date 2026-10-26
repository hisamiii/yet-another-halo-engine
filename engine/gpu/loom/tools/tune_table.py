"""The autotuner's table for a model (engine/tune/tune.py), as the emitters read it.

YAH_TILES=<file.json> selects a table explicitly, YAH_TILES= (empty) selects none; otherwise the model's checked-in
table engine/tune/tables/<model file name without .gguf>.json is used when it exists.
"""
import json
import os

TABLES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tune", "tables"))


def path(model):
    if "YAH_TILES" in os.environ:
        return os.environ["YAH_TILES"] or None
    p = os.path.join(TABLES, os.path.splitext(os.path.basename(model))[0] + ".json")
    return p if os.path.exists(p) else None


def load(model):
    p = path(model)
    return json.load(open(p)) if p else {}
