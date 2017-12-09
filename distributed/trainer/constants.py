batch_size = 128
alpha_channels = 3
width = 16
height = 16

filter_height = 5
filter_width = filter_height
convolutional_channels = 6
convolutional_skip = 2
pool_skip = 2
# from get_bottleneck_data import TOP_CLASSES
# len(TOP_CLASSES)
num_targets = 1623 + 1
learning_rate = 0.0001
beta1 = 0.99 # Default 0.9
beta2 = 0.999 # Default 0.999
epsilon = 1e-8 # Default 1e-8
num_steps = 5000

num_bottlenecks = 1001
