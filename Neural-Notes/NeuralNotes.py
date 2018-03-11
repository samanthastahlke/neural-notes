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
    rbmNet.LoadTrainingSet(appData.trainDataDirectory)
    return

#Setup global Tkinter handlers.
mainUI.tkRoot.protocol("WM_DELETE_WINDOW", WindowCloseCallback)
mainUI.btnChooseTrainData.bind("<ButtonRelease-1>", appData.GetTrainDirectory)
mainUI.btnLoadTraining.bind("<ButtonRelease-1>", LoadTrainingSet)

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

frame.Tick()
AppMain()
Cleanup()