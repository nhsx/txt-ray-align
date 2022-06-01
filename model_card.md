# TxtRayAlign: Automated Text Descriptions from Chest Radiographs

This project contains an implementation to contrastively train image and text encoders initially taken from [here](https://github.com/Zasder3/train-CLIP) and then modified for training on [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) and evaluating on an image-to-text retrieval task. 

## Model use
### Intended use
This model is intended for use in experimenting with image-to-text retrieval tasks. 

### Out-of-scope cases
This model is not suitable to generate clinically accurate reports in a production environment. 

## Training data
Experiments were run on [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/).

## Performance and limitations
The model has been tested with `ResNet50` and `EfficientNet_B0` from the `torchvision` library and `DeCLUTR-sci-base`, `Bio_ClinicalBERT` and `DistilBERT` from the `huggingface` library and the `RN50`-version of the CLIP model found [here](https://github.com/openai/CLIP). 

The evaluation of the model was based on its training metrics as well as its performance in retrieving relevant pieces of text for a given image. The strategy for achieving the latter requires labelling all pieces of text using the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) labeller and then comparing the labels of the ground-truth text against the label of the predicted text. The retrieval metrics are therefore directly affected by the performance of the labeller. We note that the CheXpert labeller is an automatic rule-based labeller which has been superseded in performance by models such as [CheXbert](https://arxiv.org/abs/2004.09167). 

The model is sensitive to hyperparameter choices. Given the size of the model and the different possible encoder pairings, extensive hyperparameter tuning was not possible within the scope of this project. 

The model has not been tested for bias and fairness. In particular, an evaluation of model performance by age, gender and race has not been performed. Additional performance differences may also be found between patients suffering from common and rare conditions.
