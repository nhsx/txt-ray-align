# TxtRayAlign: Automated Text Descriptions from Chest Radiographs
## NHSX Analytics Unit - PhD Internship Project

### About the project

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)

[[Model card]](model_card.md) [[Project Reports]](/report)

TxtRayAlign holds code to contrastively train and align text and image encoders and evaluate them on an image-to-text retrieval task using [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/). This model is not suitable to generate clinically accurate reports in a production environment.

This work has been conducted as part of the NHSX Analytics Unit PhD internship project "Automated Text Descriptions from Imaging" undertaken by Dekai Zhang and Sarah Hickman. This repository was mirrored and modified from: https://github.com/Zasder3/train-CLIP

Further information on the project can be found on the [project page](https://nhsx.github.io/nhsx-internship-projects/text-description-imaging/).
 
__Note:__ _No data, public or private are shared in this repository._

### Project structure
- The main code for training, generating embeddings and evaluation are in the root directory
- Supporting scripts for pre-processing MIMIC-CXR can be found in the `data` folder


### Preliminaries

#### Dependencies

All code was tested with: 
[![Python v3.8](https://img.shields.io/badge/python-v3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

See `requirements.txt` for package versions - installation in a virtual environment is recommended:
```
conda create --name env python=3.8
conda activate env
pip install -r requirements.txt
```
When training on GPU machines, the appropriate PyTorch bundle should be installed - for more info: `https://pytorch.org/get-started/locally/` 

Note the additonal required install in the requirements which can be performed with:
```
pip install git+https://github.com/openai/CLIP.git
```
#### GPU Support


Training with GPU is recommended. Single-GPU training has been tested with:
- Nvidia Tesla T4
- cuda 11.1
- Windows Server 2019

Multi-GPU training has been tested with:
- 4x Nvidia Tesla T4
- cuda 11.4
- Ubuntu 18.04
- DDP with NCCL backend

### Usage
There are three main scripts of interest. `train_finetune.py` is for training and finetuning contrastive models. `embeddings.py` takes the trained models and calculates embeddings for text or image datasets. `evaluate.ipynb` takes the embeddings and evaluates the performance of the models on an image-to-text retrieval tasks.


#### To get MIMIC ready for training

To train with mimic, we require the JPEG images and reports in TXT form. To download the JPEG files into the current directory, you may want to use the following command:
```
gsutil -m -u <PROJECTID_TO_EXPENSE_AGAINST> cp -r gs://mimic-cxr-jpg-2.0.0.physionet.org/files .
```
where you will need a Google Cloud Platform account, which gives you the ability to create a Project against which you can expense accessing the mimic files (physionet charges the requester of the data).

After unpacking you should find the images stored in patient-specific study folders, e.g., `/files/p10/p100000/s100000/img1.jpg`

As an optional but recommended step, the images can be downsized:
```
cd data
python resize.py --image_folder <PATH_TO_IMAGE_FOLDER> 
```

The reports are available in a .zip file and can be downloaded with one of the following:
```
gsutil -m -u <PROJECTID_TO_EXPENSE_AGAINST> cp -r gs://mimic-cxr-2.0.0.physionet.org/mimic-cxr-reports.zip .
wget -r -N -c -np --user <PHYSIONET_USER_ID> --ask-password https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip
```

After unpacking the reports, you should find them stored in patient-specific folders, e.g., `/files/p10/p100000/s100000.txt`

Next, we need to grab the relevant sections from each of the reports for which you can use the `create_section_files.py` script.
To create CSV files containing the impressions and findings in concatenated form, run:
```
cd data
python create_section_files.py --reports_path <ROOT_DIR_OF_REPORTS> --output_path <OUT_FOLDER> --concat
```

The output will be a series of CSV files with the study_id and extracted sections in concatenated form.

We are now ready to load everything into our DataModule!


#### To train with MIMIC
First, generate a train/val/test split using data/get_subset.py:
```
cd data
python get_subset.py \
	--image_folder <ROOT_OF_IMAGE_FOLDER> \
	--reports_folder <PATH_TO_FOLDER_WITH_SECTIONED_REPORTS_CSVs> \
	--train_fraction 0.9 \
	--test_fraction 0.5 \
	--output_folder <OUT_FOLDER>
```

The following command will start training on a single GPU (if available, else CPU) with half precision: 
```
python train_finetune.py \
	--train <TRAIN_SPLIT_CSV> \
	--val <VAL_SPLIT_CSV> \
	--precision 16 \
	--lr 1e-4 \
	--image_encoder efficientnet_b0 \
	--text_encoder distilbert \
	--add_projection \
	--embed_dim 768 \
	--use_pretrained \
	--batch_size 32 \
	--num_workers 4 \
	--shuffle \
	--devices 1 \
	--num_sentences 1 \
	--use_teacher \
	--max_epochs 50 \
	--log_every_n_steps 1 \
	--optimizer AdamW
```
For additional optional training arguments, please refer to [`args.py`](models/args.py).

#### To output embeddings with a trained model
For the retrieval task, we build a bank of text embeddings and a query set of image embeddings using a trained model loaded from a checkpoint (typically located in the logs folder created during training). To build the bank of embeddings:
```
python embeddings.py \
	--data_path <PATH_TO_TRAIN_SPLIT_CSV> \
	--val_path <PATH_TO_VAL_SPLIT_CSV> \
	--embed_type text \
	--save_as <SOME_FOLDER_NAME> \
	--chexpert_folder <PATH_TO_FOLDER_WITH_CHEXPERT_CSVs> \
	--config_file config.json
```
Note that this requires the reports to have been split into sentences which have been passed through the CheXpert labeller. 

To build the query set of image embeddings:
```
python embeddings.py \
	--data_path <PATH_TO_TEST_SPLIT_CSV> \
	--embed_type image \
	--save_as <SOME_FOLDER_NAME> \
	--chexpert_folder <PATH_TO_FOLDER_WITH_CHEXPERT_CSVs> \
	--config_file config.json
```

#### To evaluate the model 
After having saved the embeddings for a query set and a bank, the following command will retrieve `k` items for every item in an image query set from a bank of text and evaluate the overlap in the CheXpert labels.
```
python evaluate.py \
	--case_folders <LIST_OF_FOLDERNAMES> \
	--chexpert_folder <PATH_TO_FOLDER_WITH_CHEXPERT_CSVs> \
	--k 2 \
	--query_type image \
	--bank_type text
```
__Note:__ The evaluation step requires sentences labelled with the CheXpert labeller. The retrieval step can be performed independently. 


### References

This repository is largely based on:
```
@misc{cg2021trainCLIP,
  author = {Cade Gordon},
  title = {train-CLIP},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4915843},
  howpublished = {\url{https://github.com/Zasder3/train-CLIP}}
}
```

Learning transferable visual models from natural language supervision (a.k.a. CLIP)
```
@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}
```

Data-Efficient Language-Supervised Zero-Shot Learning with Self-Distillation
```
@article{cheng2021data,
  title={Data-Efficient Language-Supervised Zero-Shot Learning with Self-Distillation},
  author={Cheng, Ruizhe and Wu, Bichen and Zhang, Peizhao and Vajda, Peter and Gonzalez, Joseph E},
  journal={arXiv preprint arXiv:2104.08945},
  year={2021}
}
```

The MIMIC code repository
```
@article{johnson2018mimic,
  title={The MIMIC Code Repository: enabling reproducibility in critical care research},
  author={Johnson, Alistair E W and Stone, David J and Celi, Leo A and Pollard, Tom J},
  journal={Journal of the American Medical Informatics Association},
  volume={25},
  number={1},
  pages={32--39},
  year={2018},
  publisher={Oxford University Press}
}
```
### Roadmap

See the [open issues](https://github.com/nhsx/txt-ray-align/issues) for a list of proposed features (and known issues).

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### License

Distributed under the MIT License. _See [LICENSE](./LICENSE) for more information._

### Contact

To find out more about the [Analytics Unit](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch at [analytics-unit@nhsx.nhs.uk](mailto:analytics-unit@nhsx.nhs.uk).
