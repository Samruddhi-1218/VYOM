# tflite-check.py — uses .h5 directly, works on Windows dev machine
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

print("Loading vyom_mobilenet_lstm.h5...")
model = tf.keras.models.load_model(
    "models/vyom_mobilenet_lstm.h5",
    compile=False
)

classes = np.load("models/label_encoder_classes.npy", allow_pickle=True)
print(f"Classes ({len(classes)}): {list(classes)}")
print(f"Input  shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# Test with dummy input
dummy = np.random.rand(1, 45, 160, 160, 3).astype(np.float32)
result = model.predict(dummy, verbose=0)

print(f"\nRaw probs:  {result[0].round(4)}")
print(f"Sum:        {result[0].sum():.4f}")
print(f"Predicted:  {classes[np.argmax(result[0])]} ({result[0].max():.3f})")
print("\n✓ Model working — ready for vyom_inference.py")