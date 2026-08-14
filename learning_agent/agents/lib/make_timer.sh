#!/bin/bash
# Write a timer.sh into the repo root that the agent can query for time remaining.
# Usage: make_timer.sh <deadline_epoch> <out_path>
# The generated timer.sh reads the baked-in deadline and prints HH:MM:SS remaining
# (00:00:00 once the budget is spent), plus a machine-readable "seconds_left=" line.
set -euo pipefail
DEADLINE="$1"
OUT="$2"

cat > "$OUT" <<EOF
#!/bin/bash
# Learning Agent run timer — prints wall-clock remaining in this evaluation.
DEADLINE=${DEADLINE}
now=\$(date +%s)
left=\$(( DEADLINE - now ))
if [ "\$left" -lt 0 ]; then left=0; fi
printf 'time remaining: %02d:%02d:%02d\n' \$(( left/3600 )) \$(( (left%3600)/60 )) \$(( left%60 ))
echo "seconds_left=\$left"
EOF
chmod +x "$OUT"
