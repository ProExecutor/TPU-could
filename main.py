import tensorflow as tf

# Create a TPUClusterResolver
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')

# Connect to the TPU cluster
tf.config.experimental_connect_to_cluster(resolver)

# Initialize the TPU system
tf.tpu.experimental.initialize_tpu_system(resolver)

# List all TPU devices
print("All devices: ", tf.config.list_logical_devices('TPU'))
