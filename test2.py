import tensorflow as tf
import tensorflow.contrib.tensorrt as trt


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
batch_size=16
workspace_size= 4000000000
precision='FP16'

trt.create_inference_graph(
    input_saved_model_dir=input_saved_model_dir,
    output_saved_model_dir= output_saved_model_dir
)


trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph_def,
    outputs=output_node_name,
    max_batch_size=batch_size,
    max_workspace_size_bytes=workspace_size,
    precision_mode=precision,
    minimum_segment_size=3
)

input_graph_def =