#!/bin/bash
# Session-config helpers shared by agents/run*.sh — pure bash/awk (host machines
# are not guaranteed pyyaml). Reads the FLAT keys the runners need from a
# task-config-schema file: top-level `task:` and scalars under the `session:`
# block. This is deliberately not a YAML parser; session config must stay flat.
#
#   lab_task_known   <root> <task>     a task exists iff task_configs/<task>.yaml does
#   yaml_top     <file> <key>      top-level scalar (e.g. task)
#   lab_yaml_session <file> <key>      scalar under `session:` (scaffold/track/hours/model)
#
# Values are stripped of trailing comments and surrounding quotes; missing keys
# print nothing (callers apply their own defaults).

lab_task_known() {
    [ -f "$1/task_configs/$2.yaml" ]
}

yaml_top() {
    awk -v k="$2" '
        $0 ~ "^"k":" {
            sub("^"k":[[:space:]]*", "");
            sub(/[[:space:]]*#.*$/, "");
            gsub(/^["'\'']|["'\'']$/, "");
            print; exit
        }' "$1"
}

lab_yaml_session() {
    awk -v k="$2" '
        /^session:/      { insec = 1; next }
        insec && /^[^ ]/ { insec = 0 }
        insec && $1 == k":" {
            sub(/^[[:space:]]*[a-zA-Z0-9_]+:[[:space:]]*/, "");
            sub(/[[:space:]]*#.*$/, "");
            gsub(/^["'\'']|["'\'']$/, "");
            print; exit
        }' "$1"
}
