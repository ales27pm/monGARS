
import tensorflow as tf

if __name__ == "__main__":
    # Simulate TensorFlow computation
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Running TensorFlow on GPU...")
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        c = tf.matmul(a, b)
        print("TensorFlow GPU computation successful.")
    else:
        print("No GPUs detected. Simulating CPU computation.")
