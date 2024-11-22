
import tensorflow as tf

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {gpus}")
    else:
        print("No GPUs detected. Running on CPU.")
