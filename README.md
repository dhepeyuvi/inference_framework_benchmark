# Inference framework benchmark

First pre-print:
https://arxiv.org/pdf/2210.04323.pdf

# Running Order
export xargs, Pythonpath, Activate the Environment then run the files in the following order
1) generate_tf_models
2) convert.sh or run.sh to generate onnx and openvino files
3) from_SavedModel_to_trt.py to get trt files
4) inference_tf.py to run inference of tf_xla
5) inference_trt.py to run inference on trt
