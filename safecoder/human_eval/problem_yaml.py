"""
YAML schema definitions used by the functional evaluation scripts.

Note: yamlize relies on ruamel.yaml's RoundTripLoader internally. Some ruamel.yaml
versions reference `loader.max_depth` during composition, but the RoundTripLoader
class may not define it, which causes:
    AttributeError: 'RoundTripLoader' object has no attribute 'max_depth'

We add a small compatibility shim here so Problem.load() works reliably across
ruamel.yaml versions.
"""

from yamlize import Object, Attribute, Sequence, StrList, Typed

try:
    # yamlize uses ruamel's RoundTripLoader; ensure it has `max_depth`.
    from ruamel.yaml.loader import RoundTripLoader

    if not hasattr(RoundTripLoader, "max_depth"):
        RoundTripLoader.max_depth = None
except Exception:
    # If ruamel.yaml is not installed or its internals changed, let yamlize
    # raise the import/load error naturally when Problem.load() is called.
    pass

class Problem(Object):
    name = Attribute(type=str)
    language = Attribute(type=str)
    prompt = Attribute(type=str)
    tests = Attribute(type=str)
    completions = Attribute(type=StrList)
    stop_tokens = Attribute(type=StrList)

class Result(Object):
    program = Attribute(type=str)
    stdout = Attribute(type=str)
    stderr = Attribute(type=str)
    exit_code = Attribute(type=int)
    status = Attribute(type=str)
    timestamp = Attribute(type=int, default=0)

class ResultList(Sequence):
    item_type = Result

class TestResults(Object):
    name = Attribute(type=str)
    language = Attribute(type=str)
    results = Attribute(type=ResultList)
