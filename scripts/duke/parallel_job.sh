#!/bin/bash
set -euo pipefail
task=${1:-completion}
case "$task" in
    completion|synthesis) ;;
    *) echo "Usage: bash scripts/duke/parallel_job.sh [completion|synthesis]" >&2; exit 2 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for number in {1..3}; do
    sbatch --job-name="duke_${task}_exp${number}" \
        scripts/duke/job.sh "configs/duke/exp/${task}_exp${number}.yaml"
done
