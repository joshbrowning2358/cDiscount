Repo to store my code for the cDiscount Kaggle Competition.

Process:
- The data from Kaggle comes in .bson files.  Run store_data_as_images.py to save image files.
- Run images_to_bottlenecks.sh to convert the image files into "bottlenecks" (the last layer of the AlexNet deep net for image recognition).
- Run create_chunks.py to convert the bottleneck files into 10 files with all the bottleneck data (for faster loading when running TensorFlow).
- Run distributed/train_on_gcp.sh to run the final training on GCP.  The compute engines need to be running, as per this tutorial: https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine
