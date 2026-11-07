"""hapadapter.analysis -- scoring, ISM and figure tools that run on a FINE-TUNED checkpoint

Every module here is a command-line entry point:

    python -m hapadapter.analysis.<module> --help

Nothing is imported at package level; these modules pull optional third-party packages, which can be installed via:
    
    pip install hapadapter[analysis]     # figures
    pip install hapadapter[model]        # anything that loads a checkpoint

Modules:
  build_asb_ism_windows   twin ref/alt or haplotype ISM windows, leak-filtered
"""