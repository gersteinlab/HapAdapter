#!/usr/bin/env python
"""
hapadapter.trunk.load_pretrained_trunk -- 
write a trunk directory holding only the PRETRAINED encoder weights (before fine-tuning)
"""
import argparse, json, os, shutil, sys

ENCODER_FIELDS = ("model_name_or_path", "cache_dir", "pooling_mode", "center_pool_width",
                  "model_max_length", "head_activation", "head_dropout")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--like", required=True,
                    help="run_config.json of a REAL fine-tuned trunk (or its directory). Its "
                         "encoder settings are copied so the control differs only in weights.")
    ap.add_argument("--backbone", default=None,
                    help="override model_name_or_path, e.g. a local DNABERT-2-117M directory "
                         "when the path recorded in --like has moved")
    ap.add_argument("--out", required=True, help="directory to create")
    ap.add_argument("--device", default="cpu",
                    help="only affects where the model is built; the weights are saved on CPU")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    from hapadapter.trunk.loading import load_run_config
    from hapadapter.trunk.model import FunctionalFinetuneForSequencePrediction
    import torch

    src = a.like if os.path.isdir(a.like) else os.path.dirname(os.path.abspath(a.like))
    rc = load_run_config(src)

    missing = [f for f in ("model_name_or_path", "pooling_mode") if not rc.get(f)]
    if missing and not (a.backbone and missing == ["model_name_or_path"]):
        raise SystemExit(f"{src}/run_config.json records no {', '.join(missing)}; cannot "
                         f"reproduce the encoder it was built with.")

    backbone = a.backbone or rc["model_name_or_path"]
    if os.path.exists(a.out) and not a.overwrite:
        raise SystemExit(f"{a.out} exists; pass --overwrite to replace it.")

    model = FunctionalFinetuneForSequencePrediction(
        model_name_or_path=backbone,
        cache_dir=rc.get("cache_dir"),
        pooling_mode=rc.get("pooling_mode", "center_mean"),
        center_pool_width=int(rc.get("center_pool_width", 5)),
        head_num_layers=1, head_hidden_size=-1,
        head_activation=rc.get("head_activation", "gelu"),
        head_dropout=float(rc.get("head_dropout", 0.1)),
        task="regression", num_labels=1)

    model.eval()

    sd = {k: v for k, v in model.state_dict().items() if k.split(".")[0] == "backbone"}
    if not sd:
        raise SystemExit(f"the model built from {backbone} exposes no 'backbone.*' parameters, "
                         f"so neither init_backbone_from() nor loading.py could read it back.")

    os.makedirs(a.out, exist_ok=True)
    wpath = os.path.join(a.out, "pytorch_model.bin")
    tmp = wpath + ".tmp"
    torch.save({k: v.cpu() for k, v in sd.items()}, tmp)
    os.replace(tmp, wpath)

    cfg = {f: rc[f] for f in ENCODER_FIELDS if f in rc}
    cfg["model_name_or_path"] = backbone
    cfg.update(task="regression", num_labels=1,
               encoder_weights="pretrained",
               trunk_finetuned=False,
               like=os.path.abspath(src))
    cpath = os.path.join(a.out, "run_config.json")
    ctmp = cpath + ".tmp"
    with open(ctmp, "w") as fh:
        json.dump(cfg, fh, indent=2)
    os.replace(ctmp, cpath)

    mb = os.path.getsize(wpath) / 1e6
    print(f"[write] {a.out}")
    print(f"        {len(sd)} backbone tensors ({mb:.0f} MB) at PRETRAINED weights")
    print(f"        backbone      {backbone}")
    print(f"        pooling       {cfg.get('pooling_mode')} width {cfg.get('center_pool_width')}"
          f" | model_max_length {cfg.get('model_max_length')}")
    print(f"        copied from   {src}")
    print(f"\n  fit the control head with the ORDINARY command, one argument changed:")
    print(f"    python -m hapadapter.fit_adapter --backend sequence \\")
    print(f"      --init_backbone_from {a.out} \\")
    print(f"      --build_dir <cell>/<build> --out_dir <cell>/control_head  ...")

if __name__ == "__main__":
    sys.exit(main())
