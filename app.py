from flask import Flask, request
from pathlib import Path
from hashlib import md5
import onnxruntime as ort
from loguru import logger
import numpy as np

app = Flask(__name__)
all_models = {}

for model_path in Path('models').glob('*.onnx'):
    name = md5(model_path.read_bytes()).hexdigest()
    all_models[name] = ort.InferenceSession(str(model_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    logger.info(f'Loaded model {model_path} as {name}')

@app.route('/v1/<model_name>/inference', methods=['POST'])
def inference(model_name):
    if model_name not in all_models:
        return dict(error=f'Model {model_name} not found'), 404

    model = all_models[model_name]
    payload = request.get_json(silent=True)
    if payload is None:
        return dict(error='Invalid payload'), 400

    inputs = {}
    for model_input in model.get_inputs():
        name = model_input.name

        if name not in payload:
            return dict(error=f'Input {name} not found'), 400

        input = payload[name]
        if 'data' not in input or 'shape' not in input:
            return dict(error=f'Input {name} is invalid'), 400
        
        np_type = None
        if model_input.type == 'tensor(float)':
            np_type = np.float32
        elif model_input.type == 'tensor(int64)':
            np_type = np.int64
        else:
            return dict(error=f'Input {name} has unsupported type {model_input.type}'), 400

        data = np.array(
            input['data'], 
            dtype=np_type
        ).reshape(input['shape'])

        inputs[name] = data

    outputs = model.run(None, inputs)

    return dict(
        type="float32",
        shape=outputs[0].shape,
        data=outputs[0].flatten().tolist()
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6873, debug=True, threaded=True)
