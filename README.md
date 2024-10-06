# INFOMAIR Project - Group 25

This project is a chatbot-based system designed to assist users in finding restaurant recommendations based on preferences. It uses a combination of machine learning models, text processing, and a state-driven dialog management system.

## Setup and Installation

To get started with the INFOMAIR Project, follow these steps:

### Prerequisites

Ensure you have Python 3.10 installed. It's recommended to use Anaconda to set up the environment.

1. Clone the repository from the specified branch:

    ```sh
    git clone -b part1_final https://github.com/Cem-Kaya/INFOMAIR_25.git
    cd INFOMAIR_25
    ```

2. Create and activate the Anaconda environment using `environment.yml`:

    ```sh
    conda env create -f environment.yml
    conda activate infomair_env
    ```


3. **NVIDIA GPU Setup**:
   - If you have an NVIDIA GPU, make sure you have the correct version of PyTorch installed that supports CUDA. The version that has been tested is CUDA 12.4 with cuDNN 8.7.
   - Ensure that CUDA Toolkit 12.4 and cuDNN 8.7 are installed on your system. Refer to the [NVIDIA documentation](https://docs.nvidia.com/) for installation details.
   - Use the following command to install PyTorch with CUDA support:

     ```sh
     conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
     ```

4. **CPU Setup**:
   - If you do not have an NVIDIA GPU, PyTorch can still be installed to work on CPU, though it will be significantly slower.

### Training the Model

To train the classifier model used for detecting dialog actions, run:

```sh
python train_models.py
 ```


### Running the Chatbot

```sh
python main.py
```

## Configuration
The chatbot's behavior is controlled by config.json. Here are some of the configurable options:

Model Selection: Specify which pre-trained model to use by modifying the "model" field in config.json.

Text-to-Speech (TTS): Enable or disable TTS support.

Automatic Speech Recognition (ASR): Enable ASR to allow voice interaction with the chatbot. 

Delay Responses: Add a delay to simulate a more natural conversation flow.

Casual Mode: Adjust the chatbot's tone for a more casual conversation.

## Debugging


Set the DEBUG flag in main.py to True to see detailed debug output during conversation.
```python
DEBUG = True
```

