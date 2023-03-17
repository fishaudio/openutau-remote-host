# OpenUTAU remote ONNX hosting

This repository hosts the ONNX models for DiffSinger OpenUTAU. 
You should put *.onnx files in the `models` directory, after you start the server, you will see their md5 hash in the console.

## Usage
Set your `acoustic` in `dsconfig.yaml` to `http://${ADDRESS}/v1/${md5}/inference`.

To start the server, run `python3 app.py` in the root directory of this repository.  
There is also a docker image for this server, but it is not tested yet.
