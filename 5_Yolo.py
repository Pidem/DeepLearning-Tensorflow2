import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

msgs = []
inp = np.random.random_sample([1, 224, 224, 3]).astype(np.float32)
inpconst = tf.constant(inp)


def run_and_time(saved_model_dir, ref_result=None):
  """Helper method to measure the running time of a SavedModel."""
  NUM_RUNS = 100
  root = tf.saved_model.load(saved_model_dir)
  concrete_func = root.signatures["serving_default"]
  result = None
  for _ in range(2):  # warm up
    concrete_func(input_1=inpconst)

  start_time = datetime.datetime.now()
  for i in range(NUM_RUNS):
    result = concrete_func(input_1=inpconst)
  end_time = datetime.datetime.now()

  elapsed = end_time - start_time
  result = result[result.keys()[0]]

  msgs.append("------> time for %d runs: %s" % (NUM_RUNS, str(elapsed)))
  if ref_result is not None:
    msgs.append(
        "------> max diff: %s" % str(np.max(np.abs(result - ref_result))))
  return result


# Save the original Keras model.
saved_model_dir = "./tmp/mobilenet.original"
mobilenet = tf.keras.applications.MobileNet()
tf.saved_model.save(mobilenet, saved_model_dir)

# Convert the SavedModel using TF-TRT
converter = trt.TrtGraphConverter(
    input_saved_model_dir=saved_model_dir,
    precision_mode="FP16",
    is_dynamic_op=True)
converter.convert()


saved_model_dir_trt = "./tmp/mobilenet.trt"
converter.save(saved_model_dir_trt)

# Measure the performance.
ref_result = run_and_time(saved_model_dir)
run_and_time(saved_model_dir_trt, ref_result)
for m in msgs:
  print(m)

