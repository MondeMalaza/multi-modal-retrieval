# Multi-Modal-Image-Retrieval
A multi-modal image retrieval system for the Standard Bank Group Assessment for the Specialist, Artificial Intelligence role.
The system uses CLIP (Contrastive Language-Image Pretraining), an open-source model developed by OpenAI to retrieve images matching the user's descriptive prompts, from a dataset of 5 541 images.
We only randomly sample 500 images from the given dataset. 
The system includes an interactive streamlit user-interface.

## Requirements
You need to have Python and Pip installed on your machine. 

Python v3.9+

Pip v24+

## Project setup for local dev
The script can be run on local using VSCode or any other code editor that allows python scripts. 

1. Clone project to your local environment using git
   
   `$ git clone https://github.com/MondeMalaza/Multi-Modal-Image-Retrieval.git`

2. Go to the project directory and install the following libraries
   
   `$ cd multi-modal retrieval system`
   
   `$ pip install pytorch torchvision torchaudio transformers pillow faiss-cpu streamlit numpy random`
3. Ensure that you install CLIP from github and not the default one as this is the one from openAI

   `$ pip install git+https://github.com/CLIP.get`
   
## How to run the system

After installing all the required libraries, open the terminal in the same project directory and run the following command

`$ python application.py`
## Project Structure
