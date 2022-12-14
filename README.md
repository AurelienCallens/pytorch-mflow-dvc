Including MLOps in pytorch project: Classification of sea animals images with pytorch
==============================

## Overview of the project

This deep learning project aims to classify images of 19 different sea animals with pretrained MobileNetv3 pytorch model. The dataset comes from kaggle: https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste/code

This test project had several objectives:
   - Learn how to use pytorch (pytorch-lightning) for computer vision 
   - Include MLOps tools into my workflow such as:
      + DVC for data version control and pipeline
      + MLflow for experiment tracking 

In this project, I tracked each step of the DVC pipeline by nesting Mlflow runs. This was made possible by (this awesome post)[https://www.sicara.fr/blog-technique/dvc-pipeline-runs-mlflow] that provides clear explanations and code on how to track dvc pipelines with Mlflow.


## Creating virtual environment and installing dependencies

1. Download or clone this repository at the location of your choice

2. Create a virtual environment with **python 3.9.15**:

    ```
    conda create --name [env name] python=3.9.15
    ```

3. Activate your virtual environment and download the required dependencies:

    ```
    conda activate [env name]
    conda install --file requirements.txt
    ```
   
4. Download the data from kaggle (https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste/code) and extract the data in `data/raw/`.

## The deep learning pipeline

### General workflow

In progress ... 
 

### Run the pipeline

The steps of the pipeline are contained in `dvc.yaml` and the parameters conditioning the pipeline are in  `params.yaml`. 

Before running the pipeline, we must create an mlflow experiment with the command:

```
mlflow experiments create -n [name of experiment]
```

Then we can run the pipeline and track its progress with mlflow run: 

```
make run_pipeline RUN_NAME=[name_of_your_run]
```

Once the run is complete, we can change some parameters and run the pipeline once again. By creating a DVC pipeline, we run only the steps of the pipeline that have changed, which can save a lot of computing time. 

After a few runs we can access the mlflow dashboard to investigate the performances of our algorithm depending on the parameters we track: 

```
mlflow ui
```

### Examples of output

![Mlflow dashboard](/notebooks/mlflow_ui.png)

Here we can see the tracking results with the Mlflow dashboard with: 


1. First run where all the steps are executed and tracked by mlflow.

2. Names of the file/step executed

3. Names of the run (`[name_of_your_run]` in the make command. 

4. Metrics for each run




