'''
NEURALNOTES.PY
This is the script to run if you want to make some music!

Other scripts:
RBMNet is the home of the neural net.
GUIWrapper wraps some Tkinter functionality and contains all the GUI code, as well as some basic app management.
MidiWrapper contains utilities for reading/writing MIDI files.

Acknowledgements:

Our work is based on the project from this article:
http://danshiebler.com/2016-08-10-musical-tensorflow-part-one-the-rbm/
You can find the source code from that project here:
https://github.com/dshieble/Music_RBM

Additionally, we referenced these articles to help understand how RBMs work:
https://medium.com/@oktaybahceci/generate-music-with-tensorflow-midis-4bf928a35c3a
https://deeplearning4j.org/restrictedboltzmannmachine

And, of course, the Tensorflow documentation:
https://www.tensorflow.org/api_docs/python/
'''

import RBMNet as rbm
import MidiWrapper as nn_midi
import GUIWrapper as gui
import time

#Initialize core application objects.
MAX_FRAME_RATE = 60

frame = gui.FrameMgr(MAX_FRAME_RATE)
appData = gui.AppData()
mainUI = gui.MainUI()

midiUtil = nn_midi.NNMidiUtility()
rbmNet = rbm.RBMNet(midiUtil)

#Global Tkinter callbacks.
def WindowCloseCallback():
    mainUI.running = False
    return

def LoadTrainingSet(event):
    mainUI.SetTrainStatus("Loading dataset...")

    #Confirm the number of timesteps.
    try:
        tmpTimesteps = int(mainUI.tTxtTimesteps.get())
        rbm.DEFAULT_TIMESTEPS = tmpTimesteps
    except Exception:
        mainUI.tTxtTimesteps.delete(0, 'end')
        mainUI.tTxtTimesteps.insert(0, rbm.DEFAULT_TIMESTEPS)

    loadSuccess = rbmNet.LoadTrainingSet(appData.trainDataDirectory)
    mainUI.SetTrainStatus("Dataset loaded" if loadSuccess else "Dataset loading failed")

    return

def GetModelSaveDirectory(event):
    appData.GetModelSaveDirectory(event)

    if appData.modelSaveDirectory is not None:
        mainUI.tTogSave.configure(text="Save model to " + appData.modelSaveDirectory)
    else:
        mainUI.tTogSave.configure(text="Save model to ---")

    return

def GetModelLoadDirectory(event):
    appData.GetModelLoadDirectory(event)
    return

def GetSampleSaveDirectory(event):
    appData.GetSampleSaveDirectory(event)

    if appData.sampleSaveDirectory is not None:
        mainUI.gLblSaveDir.configure(text="Saving samples to " + appData.sampleSaveDirectory)
    else:
        mainUI.gLblSaveDir.configure(text="Saving samples to " + rbm.SAMPLE_LOC)

    return

def TriggerGen(event):
    mainUI.SetGenStatus("Generating samples...")

    #Extract info from our generation parameter fields.
    try:
        tmpSamples = int(mainUI.gTxtSamples.get())
        rbmNet.genSample = tmpSamples
    except Exception:
        mainUI.gTxtSamples.delete(0, 'end')
        mainUI.gTxtSamples.insert(0, rbmNet.genSample)

    try:
        tmpTimescale = int(mainUI.gTxtTimescale.get())
        midiUtil.tickScale = tmpTimescale
    except Exception:
        mainUI.gTxtTimescale.delete(0, 'end')
        mainUI.gTxtSamples.insert(0, midiUtil.tickScale)

    genResult = rbmNet.Generate(event, appData.modelLoadDirectory, appData.sampleSaveDirectory)
    mainUI.SetGenStatus("Finished generating samples" if genResult else "Sample generation failed")
    return

def TriggerTrain(event):
    mainUI.SetTrainStatus("Training...")

    #Extract info from our training parameter fields.
    try:
        tmpEpochs = int(mainUI.tTxtEpochs.get())
        rbm.DEFAULT_EPOCHS = tmpEpochs
    except Exception:
        mainUI.tTxtEpochs.delete(0, 'end')
        mainUI.tTxtEpochs.insert(0, rbm.DEFAULT_EPOCHS)

    try:
        tmpLearning = float(mainUI.tTxtLearn.get())
        rbm.DEFAULT_LEARNRATE = tmpLearning
    except Exception:
        mainUI.tTxtLearn.delete(0, 'end')
        mainUI.tTxtLearn.insert(0, rbm.DEFAULT_LEARNRATE)

    try:
        #For timesteps, if the number has increased, we need to re-load
        #training data.
        tmpTimesteps = int(mainUI.tTxtTimesteps.get())
        if tmpTimesteps != rbm.DEFAULT_TIMESTEPS :
            LoadTrainingSet(event)
    except Exception:
        pass

    try:
        tmpNodes = int(mainUI.tTxtNodes.get())
        rbm.DEFAULT_HNODES = tmpNodes
    except Exception:
        mainUI.tTxtNodes.delete(0, 'end')
        mainUI.tTxtNodes.insert(0, rbm.DEFAULT_HNODES)

    rbmNet.InitNNParameters()
    trainResult = rbmNet.Train(event, appData.modelSaveDirectory if mainUI.saveModel.get() else None)
    mainUI.SetTrainStatus("Training complete" if trainResult else "Training failed")

    if trainResult:
        mainUI.SetGenStatus("Ready")

    return

#Setup global Tkinter handlers.
mainUI.tkRoot.protocol("WM_DELETE_WINDOW", WindowCloseCallback)
mainUI.tBtnChooseData.bind("<ButtonRelease-1>", appData.GetTrainDirectory)
mainUI.tBtnLoadData.bind("<ButtonRelease-1>", LoadTrainingSet)
mainUI.tBtnChooseSave.bind("<ButtonRelease-1>", GetModelSaveDirectory)
mainUI.tBtnTrain.bind("<ButtonRelease-1>", TriggerTrain)
mainUI.gBtnChooseModel.bind("<ButtonRelease-1>", GetModelLoadDirectory)
mainUI.gBtnChooseSave.bind("<ButtonRelease-1>", GetSampleSaveDirectory)
mainUI.gBtnGen.bind("<ButtonRelease-1>", TriggerGen)

#Initialize Tkinter UI labels/elements.
#Training/model parameters.
mainUI.tTxtTimesteps.delete(0, 'end')
mainUI.tTxtTimesteps.insert(0, rbm.DEFAULT_TIMESTEPS)
mainUI.tTxtEpochs.delete(0, 'end')
mainUI.tTxtEpochs.insert(0, rbm.DEFAULT_EPOCHS)
mainUI.tTxtLearn.delete(0, 'end')
mainUI.tTxtLearn.insert(0, rbm.DEFAULT_LEARNRATE)
mainUI.tTxtNodes.delete(0, 'end')
mainUI.tTxtNodes.insert(0, rbm.DEFAULT_HNODES)

#Generation parameters.
mainUI.gTxtTimescale.delete(0, 'end')
mainUI.gTxtTimescale.insert(0, midiUtil.tickScale)
mainUI.gTxtSamples.delete(0, 'end')
mainUI.gTxtSamples.insert(0, rbmNet.genSample)
mainUI.gLblSaveDir.configure(text="Saving samples to " + rbm.SAMPLE_LOC)

#Status messages.
mainUI.SetTrainStatus("No data available")
mainUI.SetGenStatus("Ready" if rbmNet.IsTmpModelStored() else "No model available")

#Main loop.
def AppMain():

    while mainUI.running:

        frame.Tick()

        mainUI.tkRoot.update_idletasks()
        mainUI.tkRoot.update()

        if frame.TimeSinceTick() < frame.targetDelta:
            time.sleep(frame.targetDelta - frame.TimeSinceTick())

    return

#Global resource cleanup.
def Cleanup():
    mainUI.tkRoot.destroy()
    return

#Jump in to the application!
frame.Tick()
AppMain()
#Destroy all the evidence!
Cleanup()