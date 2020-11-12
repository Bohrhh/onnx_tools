# Onnx Tools

## Requirements

* python3
* onnx >= 1.6.0
* onnxruntime >= 1.1.2
* albumentations
* cv2
* tqdm
* ...

## Directory

```
├── onnx_models/ # models or something else
├── base.py # general methods for onnx models
├── simplify.py # simplify onnx models
├── calibrate.py # generate dynamic range file
├── modify.py # modify onnx models

```

## Usage

simplify your model
```
import simplify as s
help(s.main)

```

calibrate to generate dynamic file  
you should implement your own dataset
```
import calibrate as c
help(c.main)
```

modify onnx models which have specific operators
```
import modify as m

```

