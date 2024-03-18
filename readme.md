# Dynamic Spatial-Temporal Aggregation for Skeleton-Aware Sign Language Recognition

## Data preparation
The preprocessed skeleton data for NMFs-CSL, SLR500, MSASL and WLASL datasets are provided [here](https://mega.nz/folder/JytAARrA#76Ug-Khu-8Eskmrw1HhMCQ). Please be sure to follow their rules and agreements when using the preprocessed data.

For datasets (WLASL100, WLASL300, WLASL1000, WLASL2000, MLASL100, MLASL200, MLASL500, MLASL1000, SLR500, NMFs-CSL) used to train or test our model, first create a soft link. For example, for WLASL2000:
```
ln -s path_to_your_WLASL2000/WLASL2000/ ./data/WLASL2000
```
## Pretrained models
We provide the pretrained weight for our model on the WLASL2000 dataset to validate its performance in ./pretrained_models.

## Training and testing:
### Training

Conduct the following commands: 
```
mkdir save_models
```
```
python -u main.py --config config/train.yaml --device your_device_id
```

### Testing:
```
python -u main.py --config config/test.yaml --device your_device_id
```

To test your model with pretrained weights, you may modify the line 52 in ./config/test.yaml to path of your pretrained weight.

## Acknowledgements

This code is based on [SAM-SLR-v2](https://github.com/jackyjsy/SAM-SLR-v2) and SLGTformer(https://github.com/neilsong/SLGTformer). Many thanks for the authors for open sourcing their code.