# FRAUD-Detect

This is the official repository of [Washing The Unwashable : On The (Im)possibility of Fairwashing Detection](https://nips.cc/Conferences/2022/Schedule?showEvent=54741), a work published in the Thirty-Sixth Conference on Neural Information Processing Systems (NeurIPS), New Orleans, Louisiana, USA, November 28 - December 9, 2022.<br>

![BlockDiagram](FRAUD-Detect.png)

## Setup:
1. Clone source code from GitHub:

    ```bash
    git clone https://github.com/cleverhans-lab/FRAUD-Detect.git 
    ```

2. Create virtual environment (use Python 3.8):

    ```bash
    python3 -m venv FRAUDDetect_env 
    source /FRAUDDetect_env/bin/activate
    pip3 install -r requirements.txt
    ```
    
## Description
The code contains: 
1. Training black-box models
2. Fairwashing and deteting fairwashing using FRAUD-Detect
4. Evading FRAUD-Detect using an informed adversary


### Training black-box models
The `./FRAUD-Detect_code/models/` directory contains several files to train
black-box models (AdaBoost, DNN, RF, and XgBoost) from scratch, **their
architectures and all hyper-parameters** are located in the `train_models.py` 
script, though `main.sh` will train the models for all seed and generate 
labels for both the suing and test set which are identical to those used
in our paper.

Note: training the black-box models requires tensorflow. 

#### Outputs
Trained black-box models. Note that all the pretrained models are provided at `./FRAUD-Detect/FRAUD-Detect_code/models/pretrained/` directory.

### Fairwashing and detecting fairwashing using FRAUD-Detect

FRAUD-Detect detects fairwashed models by measuring the Kullback–Leibler (KL) divergence over subpopulation-wise confusion matrices of the interpretable model.

To observe the KL_confusion_matrix vs Demographic Parity
graph for 10 seeds of a given dataset, interpretable model,
and black-box model combination:

```bash
cd FRAUD-Detect_code/washing_analysis
python analysis.py --help
```

Rerun the last line with requested arguments (note, epsilons are 
optional and given as space separated floats).
Results will be generated in ``./sample_results/`` 
directory. 

The code currently supports:

- 3 datasets (Adult income, COMPAS, and Bank Marketing)
- 4 black-box models (DNN, AdaBoost, XgBoost, and Random Forest)
- 2 interpretable models (Descision Trees, Logistic Regression)


#### Outputs
* The KL divergence as a function of Demographic Parity plot for each seed with dashed lines 
showing the black-box fairness;
* Demographic Parity and KL divergence as a function of fairwashing strength plot with error shadings. 

### Evading FRAUD-Detect using an informed adversary
In this section, we investigate on whether a dishonest entity could evade \name while performing fairwashing. 
We assume an informed adversary who is aware of the FRAUD-Detect and desires to evade FRAUD-Detect while performing fairwashing. 

The ./FRAUD-Detect/FRAUD-Detect_code/quantifying_fairwashing/ directory contains codes that explore the range of fairness gap given a fixed value of fidelity and a fixed value of  KL divergence via solving the informed adversary optimization problem.

#### Outputs
A plot showing the range of Demographic Parity gap achievable by the informed adversary seeking to evade the fairwashing detector. 

## References
If you use our code, please cite the following paper:

      @InProceedings{shamsabadi2022FraudDetect,
        title = {Washing The Unwashable : On The (Im)possibility of Fairwashing Detection},
        author = {Shahin Shamsabadi, Ali and Yaghini, Mohammad and Dullerud, Natalie and Wyllie, Sierra and Aïvodji, Ulrich and Alaagib, Aisha and Gambs, Sébastien and Papernot, Nicolas},
        booktitle = {The Thirty-Sixth Conference on Neural Information Processing Systems (NeurIPS)},
        year = {2022},
        address = {New Orleans, Louisiana, USA},
        month = November 28-December 9
      }

Citations: 

https://github.com/aivodji/LaundryML (GNU General Public License v3.0)




