# lab3

## How to use this repository

1. find the corresponding folder to the platform you want to use
2. create a python virtual environment with requirements.yaml
3. run the `app.py` file in the folder

## How to add or delete a model

We have implemented a simple way to add or delete models in this repository. The models are stored in the `models` folder, and the model instances are created in the `constants.py` file.

There are two types of models in this repository:

1. **Local models**: these are models that are stored in the `models` folder.
2. **Remote models**: these are models that can use via API.

It is easy to add or delete a model in this repository, all you need to do is to modify the `constants.py` file in the corresponding folder.

There are two things you need to do:
```python
# 1. Modify the MODELS list, this will be used to display the models in the UI.
MODELS = [
    "YOLO-plain",
    "YOLO-plain-remote",
    ...
]
# 2. Modify the MODELS_INSTANCES dictionary, this will be used to create the model instances.
MODELS_INSTANCES = {
    "YOLO-plain": YOLO("models/YOLO-plain.pt"),
    "YOLO-plain-remote": APIModel("140.124.181.195", 8010, "models/YOLO-plain.pt"),
    ...
}
```

## Something you should know about remote models

Remote models are models that can be used via API. But you still need to upload the model file to the server. Then you can use the model via API.

You can use the code in `server/` to run the server. You can modify the port in the corresponding file.

