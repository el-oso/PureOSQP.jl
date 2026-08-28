#!/usr/bin/env bash
# Reap the Julia processes a struct change strands: ReTestItems workers and `jl` daemons
# both cache the old type and fail with a misleading @world MethodError until restarted.
# The editor's language server is deliberately left alone.
pkill -f 'julia.*--code-coverage' 2>/dev/null
"$HOME/.claude/bin/jl" --kill-all >/dev/null 2>&1
sleep 2
printf 'julia RSS: %.1f GB | workers: %s | daemons: %s | jetls: %s\n' \
  "$(ps -eo rss,args | grep -i '[j]ulia' | awk '{s+=$1} END {print s/1048576+0}')" \
  "$(pgrep -cf 'code-coverage' || echo 0)" \
  "$(pgrep -cf 'DaemonMode' || echo 0)" \
  "$(pgrep -cf jetls || echo 0)"
exit 0
