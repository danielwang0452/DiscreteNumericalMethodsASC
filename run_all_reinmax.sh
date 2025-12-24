#!/usr/bin/env bash
set -euo pipefail

optimizer_options=("Adam" "RAdam")
learning_rate_options=(0.001 0.0007 0.005 0.0003)
temperature_options=(0.1 0.3 0.5 0.7 1.0 1.1 1.2 1.3 1.4 1.5)
categorical_dim_options=(8 4 8 16 64 10)
latent_dim_options=(4 24 16 12 8 30)

num_pairs=${#categorical_dim_options[@]}

for optimizer in "${optimizer_options[@]}"; do
  for lr in "${learning_rate_options[@]}"; do
    for temp in "${temperature_options[@]}"; do
      for ((i=0; i<num_pairs; i++)); do
        cat_dim=${categorical_dim_options[$i]}
        lat_dim=${latent_dim_options[$i]}
        job_name="job_${optimizer}_lr${lr}_t${temp}_cat${cat_dim}_lat${lat_dim}"

        echo "Submitting $job_name"
        qsub -N "$job_name" -v \
optimizer_type="$optimizer",learning_rate="$lr",temperature="$temp",categorical_dim="$cat_dim",latent_dim="$lat_dim" \
          run_one_job.pbs
      done
    done
  done
done