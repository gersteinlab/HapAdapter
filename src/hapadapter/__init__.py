"""
hapadapter --
transfer learning protocol for genomic foundation models adapted to diploid personal epigenomes

Public names are resolved lazily (PEP 562): 
importing `hapadapter` itself pulls in no third-party package,
so `import hapadapter` works in an environment without torch, pyBigWig or pyfaidx
"""

__version__ = "0.1.0"

# Public name -> the submodule that defines it
_EXPORTS = {
    # dataset building
    "build_dataset": "data.build_inputs",
    "WindowSpec": "data.build_inputs",
    "LabelSpec": "data.build_inputs",
    "BalanceSpec": "data.build_inputs",
    "PartitionSpec": "data.build_inputs",
    "RowSource": "data.build_inputs",
    "HaplotypeCountRowSource": "data.build_inputs",
    "MultiTissuePeakRowSource": "data.build_inputs",
    # config-driven runner
    "run_from_config": "data.build_experiment",
    "load_config": "data.build_experiment",
    # personal genomes
    "PersonalGenome": "data.personal_seq",
    "parse_chain_file": "data.personal_seq",
    # model + scoring
    "FunctionalFinetuneForSequencePrediction": "trunk.model",
    "load_model_and_tokenizer": "loading",
    "run_inference": "loading",
    "build_windows": "scoring.stats",
    "load_hetsnv": "scoring.stats",
}

__all__ = sorted(_EXPORTS) + ["__version__"]

def __getattr__(name):
    """PEP 562 lazy attribute access -- import the submodule only when the name is used"""
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)

def __dir__():
    return __all__
