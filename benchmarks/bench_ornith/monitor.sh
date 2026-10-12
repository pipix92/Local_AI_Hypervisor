#!/usr/bin/env bash
# Monitor: log GPU VRAM + download progress every 60s.
LOG=/c/Users/User/Documents/ai_tui/bench_ornith/monitor.log
DL='D:\models\mradermacher\Huihui-Ornith-1.5-9B-abliterated-i1-GGUF'
for i in $(seq 1 400); do
  TS=$(date '+%H:%M:%S')
  GPU=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null)
  LS=$(powershell -NoProfile -Command "Get-ChildItem '$DL' | Select-Object -ExpandProperty Length" 2>/dev/null | tr '\n' ' ')
  LSRAW=$(powershell -NoProfile -Command "Get-ChildItem '$DL' | Select-Object -ExpandProperty Name" 2>/dev/null | tr '\n' ' ')
  echo "$TS GPU=[$GPU] FILES=[$LSRAW] BYTES=[$LS]" >> "$LOG"
  sleep 60
done
