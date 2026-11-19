#!/bin/bash
# Chấm VLM (tools/score_vlm.sh) trên tập con MODS: nguyên bản và có lọc (hộp > 50% ảnh, nhãn water/sky/shore)
cd /home/hau/WaRS; source .venv/bin/activate
D=${D:-~/Downloads/results_VLM/runs}; SEQ=$(cat configs/mods_vlm_subset.txt)
for m in ${MODELS:-vlm_paligemma2 vlm_qwen25vl}; do
  [ -f $D/$m/boxes.json ] || continue
  for v in raw filt; do
    extra=""; [ $v = filt ] && extra="--max-area-frac 0.5 --drop-labels water sky shore land"
    [ $m = vlm_owlv2 ] && extra="$extra --score-thr ${THR:-0.2}"
    python tools/boxes_to_mods.py --boxes $D/$m/boxes.json --out preds/mods_vlm/${m}_$v $extra | tail -1
    python tools/eval_mods.py --pred preds/mods_vlm/${m}_$v --mods-root ~/datasets/mods --sequences $SEQ --out-json results/mods_vlm/${m}_$v.json --stats 2>&1 | tail -1
  done
done
echo SCORE_DONE
