![W-UNETR-pipeline](images/full_pipeline.png)
- W-UNETR is based on MONAI 0.7.0 : PyTorch-based, open-source frameworks for deep learning in healthcare signals. 
(https://github.com/Project-MONAI/MONAI)
https://arxiv.org/abs/2103.10504

This is my "open-box" version if I want to modify the parameters for some particular task, while the two above are hard-coded. The monai 0.5.0 folder contains the previous versions based on the old monai version.

*******************************************************************************
## Requirements
Follow the steps in "installation_commands.txt". Installation via Anaconda and creation of a virtual env to download the python libraries and pytorch/cuda.
*******************************************************************************
## Python scripts and their function

- fetal-ecg-synthetic-database-1.0.0/generate_dataset.py: Organize the data in the folder structure (fecg_ground,mixture,mecg_ground) for the network. 

- init.py: List of options used to train the network. 

- networks.py: The W-UNETR architecture for FECG extraction testing on simulation dataset.
- networks_real.py: The W-UNETR architecture for FECG extraction testing on real dataset.


- test_simulation.py: Runs the testing on simulation dataset.
- test_real.py: Runs the testing on real dataset.


*******************************************************************************
## Usage
### Folders structure:

First run the "fetal-ecg-synthetic-database-1.0.0/generate_dataset.py" and "ADFECGDB/generate_dataset_real.py" to create organize the simulation and real data, respectively.

Then run the "fetal-ecg-synthetic-database-1.0.0/Dataset_gen2.py" and "ADFECGDB/Dataset_gen_real.py" to create the dataloader files for the simulation and real data, respectively.

Finally, download the trained simulation and real corresponding models from the following links:
-https://drive.google.com/file/d/1g3off6VSUt1_mHnofg4FCao-RN7sCvQJ/view?usp=sharing
-https://drive.google.com/file/d/1ajsOulA9xUFgwgfvx3TeZFWx20iLXRxp/view?usp=sharing
*******************************************************************************
### Results:

The following plot show results on simulation dataset:

![simulation](images/synthetic_results.png)

The following plot show results on real dataset:

![real](images/real_results.png)

### W-UNETR Notes from the authors:

W-UNETR paper with more descriptions is now publicly available. Please check for more details:
https://arxiv.org/pdf/XXXXXXX.pdf

