BUCKET="cdiscount-josh"
JOB_NAME="job_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training ${JOB_NAME} \
    --package-path trainer \
    --module-name trainer.task \
    --staging-bucket gs://${BUCKET} \
    --job-dir gs://${BUCKET}/${JOB_NAME} \
    --runtime-version 1.2 \
    --region us-central1 \
    --config config/config.yaml \
    -- \
    --data_dir gs://${BUCKET}/input \
    --output_dir gs://${BUCKET}/${JOB_NAME} \
    --train_steps 10000

