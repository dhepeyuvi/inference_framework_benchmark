import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def identity(x):
    return x

def batch_generator(data, batch_size, prepro_f):
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        batch_data = prepro_f(batch_data)
        yield i, batch_data

def load_tf_model(INPUT_TENSOR_NAME, OUTPUT_TENSOR_NAME, PB_PATH, GRAPH_NAME, config):
    batch_size = config["batch_size"]

    imgsize = config.get("imgsize", 224)

    # Load the model
    model = load_model(PB_PATH)

    # Extract input and output tensors
    input_tensor = model.get_layer(INPUT_TENSOR_NAME).input
    output_tensor = model.get_layer(OUTPUT_TENSOR_NAME).output

    # Warm up the model with dummy data
    dumb_batch_shape = [batch_size] + input_tensor.shape.as_list()[1:]
    if dumb_batch_shape[1] is None:  # This is a patch
        dumb_batch_shape[1] = imgsize
        dumb_batch_shape[2] = imgsize

    dumb_batch = np.random.uniform(0, 1, dumb_batch_shape)
    _ = model.predict(dumb_batch)

    return input_tensor, output_tensor

def get_input_output_tensor_name(pb_path,sess,graph_name="g"):
    # CATCH INPUT/OUTPUT TENSORS
    input_regex_tensor_name = "^"+graph_name+"/x:|^x:|input:"
    output_regex_tensor_name = ".*/Softmax:0"
    input_tensor_name, output_tensor_name = tensorflow_graph_parser.explore_tf_file(
        pb_path,
        input_regex_tensor_name,
        output_regex_tensor_name,
        sess=sess,
        GRAPH_NAME=graph_name,
        verbose=False)
    return input_tensor_name, output_tensor_name


class TensorflowInferenceEngine:
    def __init__(self, path, config):
        if not os.path.exists(path):
            raise ValueError(f"ERROR in TensorflowInferenceEngine: \n  --->  {path} does not exist")

        self.batch_size = config["batch_size"]
        self.gpuid = config["gpuid"]
        self.graph_name = "g" + str(np.random.uniform(0, 999999, (1,))[0].astype(np.int32))
        self.is_XLA = config["XLA"]

        if "prepro_f" in config:
            self.prepro_f = config["prepro_f"]
        else:
            self.prepro_f = identity

        # Use XLA
        if self.gpuid != -1 and self.is_XLA:
            # Check XLA is true for better performance (but slower init. speed).
            assert os.environ["TF_XLA_FLAGS"] == "--tf_xla_auto_jit=2"
            assert os.environ["XLA_FLAGS"] == "--xla_gpu_cuda_data_dir=" + os.environ["CUDA_HOME"]

        # create TF session and bind device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)
        nbgpu = 1 if self.gpuid != -1 else 0
        tfconf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                          device_count={'CPU': 1, 'GPU': nbgpu})
        if nbgpu > 0:
            tfconf.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=tfconf)

        inout = get_input_output_tensor_name(path, self.graph_name)
        self.input_tensor_name = inout[0]
        self.output_tensor_name = inout[1]
        self.FAKE_PREDICT_MODE = False  # useful to measure Python overhead

        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "True"

        self.input_tensor, self.output_tensor = load_tf_model(
            self.input_tensor_name,
            self.output_tensor_name,
            path,
            self.graph_name,
            config
        )

    def predict(self, x):
        # fast init
        pred = np.zeros((x.shape[0], self.output_tensor.shape[1]), dtype=np.float32)
        gen = batch_generator(x, self.batch_size, self.prepro_f)

        # prediction
        if self.FAKE_PREDICT_MODE:
            return pred  # useful to measure Python overhead

        for i, batch_data in gen:
            pred[i:i + self.batch_size] = self.session.run(
                self.output_tensor,
                feed_dict={self.input_tensor: batch_data})
        return pred

    def _free_gpu_memory(self):
        K.clear_session()
        if self.session is not None:
            self.session.close()
            self.session = None
        self.input_tensor = None
        self.output_tensor = None

    def is_ok(self):
        sess_ok = self.session is not None
        input_ok = self.input_tensor is not None
        output_ok = self.output_tensor is not None
        return sess_ok and input_ok and output_ok

    def __del__(self):
        self._free_gpu_memory()


if __name__ == "__main__":
    from pierrick_tools.benchmark import BENCH
    import os
    from dotenv import load_dotenv
    
    for g in [0]:
        config = {}
        config["gpuid"] = g
        config["XLA"] = True
        print(config)
        # Directory containing pb files
        models_lib_dir = "./models_lib/TF_PB"

        # List all .pb files in the directory
        pb_files = [f for f in os.listdir(models_lib_dir) if f.endswith(".pb")]

        # Check if at least one .pb file is found
        if not pb_files:
            print("No .pb files found in the directory:", models_lib_dir)
            sys.exit(1)

        # Select all available models
        selected_models = [os.path.join(models_lib_dir, model) for model in pb_files]

        # Iterate over selected models
        for model_path in selected_models:
            print("Selected Model Path:", model_path)
            # Check if 'TF_XLA_FLAGS' is set in the environment
            tf_xla_flags = os.environ.get("TF_XLA_FLAGS")
            if tf_xla_flags is not None and tf_xla_flags == "--tf_xla_auto_jit=2":
                BENCH(TensorflowInferenceEngine, model_path, config, [32])
            else:
                print("TF_XLA_FLAGS is not set or has an incorrect value.")
