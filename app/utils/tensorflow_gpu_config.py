
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

class TensorFlowGPUConfigurator:
    @staticmethod
    def configure():
        """Enable GPU memory growth and set mixed precision policy."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                set_global_policy('mixed_float16')  # Enable mixed precision for performance
                print("TensorFlow GPU configuration complete. GPUs:", gpus)
            else:
                print("No GPUs detected. Running on CPU.")
        except Exception as e:
            print(f"Error configuring TensorFlow GPU: {e}")

if __name__ == "__main__":
    TensorFlowGPUConfigurator.configure()
