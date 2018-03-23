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
    #Confirm the number of timesteps.

    try:
        tmpTimesteps = int(mainUI.tTxtTimesteps.get())
        rbm.DEFAULT_TIMESTEPS = tmpTimesteps
    except Exception:
        mainUI.tTxtTimesteps.delete(0, 'end')
        mainUI.tTxtTimesteps.insert(0, rbm.DEFAULT_TIMESTEPS)

    rbmNet.LoadTrainingSet(appData.trainDataDirectory)
    return

def GetModelSaveDirectory(event):
    appData.GetModelSaveDirectory(event)
    mainUI.tLblSave.configure(text=appData.modelSaveDirectory)
    return

def SaveModel(event):
    GetModelSaveDirectory(event)
    rbmNet.SaveModel(event, appData.modelSaveDirectory)
    return

def GetModelLoadDirectory(event):
    appData.GetModelLoadDirectory(event)
    return

def GetSampleSaveDirectory(event):
    appData.GetSampleSaveDirectory(event)
    mainUI.gLblSaveDir.configure(text="Saving samples to " + appData.sampleSaveDirectory)
    return

def TriggerGen(event):
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

    rbmNet.Generate(event, appData.modelLoadDirectory, appData.sampleSaveDirectory)
    return

def TriggerTrain(event):
    #Extract info from our training parameter fields.

    try:
        tmpEpochs = int(mainUI.tTxtEpochs.get())
        rbm.DEFAULT_EPOCHS = tmpEpochs
    except Exception:
        mainUI.tTxtEpochs.delete(0, 'end')
        mainUI.tTxtEpochs.insert(0, rbm.DEFAULT_EPOCHS)

    try:
        tmpLearning = int(mainUI.tTxtLearn.get())
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
    rbmNet.Train(event, appData.modelSaveDirectory)
    return

#Setup global Tkinter handlers.
mainUI.tkRoot.protocol("WM_DELETE_WINDOW", WindowCloseCallback)
mainUI.btnSaveModel.bind("<ButtonRelease-1>", SaveModel)
mainUI.tBtnChooseData.bind("<ButtonRelease-1>", appData.GetTrainDirectory)
mainUI.tBtnLoadData.bind("<ButtonRelease-1>", LoadTrainingSet)
mainUI.tBtnChooseSave.bind("<ButtonRelease-1>", GetModelSaveDirectory)
mainUI.tBtnTrain.bind("<ButtonRelease-1>", TriggerTrain)
mainUI.gBtnChooseModel.bind("<ButtonRelease-1>", GetModelLoadDirectory)
mainUI.gBtnChooseSave.bind("<ButtonRelease-1>", GetSampleSaveDirectory)
mainUI.gBtnGen.bind("<ButtonRelease-1>", TriggerGen)

#mainUI.btnTrain.bind("<ButtonRelease-1>", rbmNet.Train)
#mainUI.btnGenerate.bind("<ButtonRelease-1>", rbmNet.Generate)

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

#Main loop.
def AppMain():
#
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

frame.Tick()
AppMain()
Cleanup()