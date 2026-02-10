#pip install --force-reinstall --user ./faster_muons_torch
aux_ld_preload="$LD_PRELOAD"
export LD_PRELOAD=""
pip install --force-reinstall --user ./faster_muons_torch
export LD_PRELOAD="$aux_ld_preload"
unset aux_ld_preload
