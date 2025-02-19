# Multi-Modal-Image-Retrieval
A multi-modal image retrieval system for the Standard Bank Group Assessment for the Specialist, Artificial Intelligence role.
The system uses CLIP (Contrastive Language-Image Pretraining), an open-source model developed by OpenAI to retrieve images matching the user's descriptive prompts, from a dataset of 5 541 images.
We only randomly sample 500 images from the given dataset. 
The system includes an interactive Flas-based Web UI.

## Requirements
You need to have Python and Pip installed on your machine. 

Python v3.9+

Pip v24+

## Project setup for local dev
The script can be run on local using VSCode or any other code editor that allows python scripts. 

1. Clone project to your local environment using git
   
   `$ git clone https://github.com/MondeMalaza/Multi-Modal-Image-Retrieval.git`

2. Go to the project directory 
   
   `$ cd multi-modal-retrieval`

   ...and install the following libraries
   
   `$ pip install kaggle flask pytorch torchvision torchaudio transformers pillow faiss-cpu streamlit numpy random`
3. Ensure that you install CLIP from github and not the default one as this is the one from openAI

   `$ pip install git+https://github.com/CLIP.get`

4. Setup your Kaggle API so you can be able to use the dataset found at `https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset/data?select=test_data_v2`

   Get your Kaggle API key from your Kaggle settings.
   
   Save it as `~/.kaggle/kaggle.json` on your local machine. 
   
## How to run the system

After installing all the required libraries, open the terminal in the same project directory. 

1. To run the retrieval system independent of the Web-Based UI, run the following command

   `python app.py`

2. To run the Web-Based UI image retrieval system, run the following command

   `$ python web_app.py --web`

   Open `http://127.0.0.1:5000` in your browser
## Project Structure

multi-modal-retrieval/

│── test_data_v2/

│── models/

│── templates/

│── src/

│      │── __init__.py

│      │── preprocess.py

│      │── model.py

│      │── index.py

│      │── retrieval.py

│      │── web_app.py

│── tests/

│── requirements.txt

│── README.md               

