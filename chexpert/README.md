# CheXpert Modifications 
## Sentence-level Labels

For the evaluation aspect of this setup, we have used the CheXpert labeller ([stanfordmlgroup/chexpert-labeler](https://github.com/stanfordmlgroup/chexpert-labeler)) but modified it to output both the labels plus the specific sentences within a report they are identfied within.  Please refer to the instalation instructions in the source repository for setup of the labeller.

Included in this folder are the modifications to the original codebase at [commit 4629609](https://github.com/stanfordmlgroup/chexpert-labeler/tree/4629609647d027b1dc9d4f340f5d3c03b4fb4e4f) which allow for this extended output, a stripped back `working_environment.yml`, and an example notebook (`chexpert_sentence_process.ipynb`) for running this approach across a local copy of the raw MIMIC-CXR reports.

Outputs from this processing pipeline come in the form:

| mimic_id  | report   | cat      | vals     | sents    | 
| --------- | -------- | -------- | -------- | -------- |
| s12345678 | report_cont | Fracture       | 0      | corresp_sent |
| s12345678 | report_cont | Pneumothorax | 1     | corresp_sent     |
| s12345679 | report_cont | Lung Lesion | -1      | corresp_sent     |
| s12345680 | report_cont | Edema | 0     | corresp_sent     |
| ... | ... | ... | ... | ... |

where the `vals` are defined as in the original labeller.

**Warning:**  Applying this pipeline across the whole of MIMIC-CXR takes a significant amount of time to complete.

The original CheXpert paper can be found:
```
@inproceedings{irvin2019chexpert,
  title={CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison},
  author={Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Haghgoo, Behzad and Ball, Robyn and Shpanskaya, Katie and others},
  booktitle={Thirty-Third AAAI Conference on Artificial Intelligence},
  year={2019}
}
```