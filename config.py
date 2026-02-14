import platform
import tensorflow as tf


def get_device(use_apple_gpu=False):
    gpus = tf.config.list_physical_devices("GPU")
    is_apple_silicon = (
        platform.system() == "Darwin"
        and platform.machine().lower() in {"arm64", "aarch64"}
    )

    if gpus and is_apple_silicon and not use_apple_gpu:
        try:
            # nasconde la GPU
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
        print("Device: cpu (Apple GPU disabilitata)")
        return "/CPU:0"

    if gpus:
        print("Device: gpu")
        return "/GPU:0"

    print("Device: cpu")
    return "/CPU:0"