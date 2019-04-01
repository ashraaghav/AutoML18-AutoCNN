# ML4AAD WS18/19 Final Project: AutoCNN

This project is a final submission to the AutoML / ML4AAD course. The goal is to optimize convolutional networks, built using pytorch, on [KMNIST](https://github.com/rois-codh/kmnist#kuzushiji-mnist-1) and [K49](https://github.com/rois-codh/kmnist#kuzushiji-49) datasets. 
The hardware restriction was to use only 1 CPU core for upto 24 hours to get the best architecture.

The `src` folder contains the code files required for the task and also 2 shell scripts that were used to generate results for KMNIST and K49 respectively. 

The `results` folder contains all the results from BOHB runs, including (partial) CAVE outputs<sup>[1]</sup>.

### Approach

This project formulates Neural Architecture Search as a hyper-parameter optimization problem and leverages a **Multi-Multi-Fidelity** approach to achieve the solution:
* Fidelity #1: training steps, from [BOHB](https://arxiv.org/abs/1807.01774)
* Fidelity #2: Subset selection, from [FaBOLAS](https://arxiv.org/abs/1605.07079)

Additionally, early termination was also used to speed up the optimization. 

In order to keep things comparable to the baseline, no skip connections were used. 
[CAVE](https://github.com/automl/CAVE) was later used to analyze the optimization runs.

#### Results

| Dataset | Keras Simple CNN [benchmark](https://github.com/rois-codh/kmnist#benchmarks--results-) | AutoCNN |
|---|---|---|
| KMNIST | 94.63% | 96.79% |
| K49 | 89.36% | 94.86% |

----
*[1]: The partial CAVE outputs do not include "Parameter Importance" because of issues with forbidden clauses in configuration space. This issue was raised but fixed too late, hence could not include them in the results.*