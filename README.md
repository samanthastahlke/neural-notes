# Neural Notes
an Ominous Games project

## Team
Josh Bellyk - Lead Artist and Digital Music Wizard  
Owen Meier - Lead Designer and Data Wrangler  
Samantha Stahlke - Lead Programmer and Tensorflow Apprentice  

## About
Neural Notes is an exploration of machine learning for digital music generation. We are creating an experimental tool for game developers and composers to create music samples with AI as part of the creative process.

Our work for this project was based on the RBM Music project by Daniel Shieble (https://github.com/dshieble/Music_RBM). 

You can view a video demo of our results [here](https://www.youtube.com/watch?v=xgecXJzwZ4k).

## Installation

To use Neural Notes, you'll need Python, as well as the following dependencies (it is recommended that you use pip to install):  
  
[Tensorflow](https://www.tensorflow.org/install/)  
tkinter (included with latest standard Python installation)  
numpy  
pandas
msgpack-python  
glob2 (used for directory crawling)  
tqdm (used for progress bars)  
Python Midi (install via "pip install git+https://github.com/vishnubob/python-midi@feature/python3")  
Pillow  

You can run the code via console simply by navigating to the Neural-Notes folder and running `python NeuralNotes.py`. You can also open the included scripts in an IDE such as PyCharm, if you wish (recommended option for those who wish to noodle with the inner workings).  

## How to Use Neural Notes

The application has two main modes - training and generation. Both can be accessed from the main menu.

### In Training Mode:  
1. Select "Choose training folder..." and select a folder containing MIDI files to train the network.  
2. You can enter custom values for epochs, learning rate, hidden nodes, and timesteps on this page. "Timesteps" affects the length of generated compositions - larger values will yield longer samples, but will inflate training time. Very large values should also be used in conjunction with a larger hidden layer size. The number of epochs should generally be inversely proportional to the size of the training set used - too few, and you'll have noisy key-slamming in your samples. Too many, and you'll end up with an overtrained network that tends towards silence.  
3. Select "Choose model save directory..." and select an EMPTY folder to store the model. To prevent data loss, models will not overwrite non-empty directories. Check the "Save model to..." box if you wish to persist the model beyond the cache (the application will store the last trained model in data/tmp_model).  
4. Hit "Load Training Data" to process training data from the selected folder. Check the console for progress.  
5. Hit "Train!" to build the model and train it based on loaded data. Check the console for progress.  
  
### In Generation Mode:  
1. Select "Load Model..." and choose a folder containing the model you wish to load. If no directory is selected, the application will check the tmp_model cache.  
2. You can enter custom values for the number of samples to generate and the MIDI "tick scale" (this changes the speed of the final composition, larger values create slower songs). The number of timesteps (length) to generate will be dependent on the model that is loaded.
3. Select "Choose sample save directory..." to pick a folder where generated compositions should be saved.  
4. Hit "Generate!" to generate samples. This may take longer if the program has just been loaded.  

### General Notes:  
Check the console if the program isn't behaving as expected - this may be due to your Tensorflow installation or invalid directory selection.  
  
Contact samanthastahlke@gmail.com if you have any questions or require technical support.  
  
This applet has been tested on Windows 10 with Python 3.6.3, the native Python GPU distribution of Tensorflow, and the dependencies listed above ONLY. Apologies for any difficulties with compatibility, please refer inquiries to our technical support contact listed above. 
