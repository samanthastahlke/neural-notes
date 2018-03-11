from tkinter import *
from tkinter import filedialog
import time

class FrameMgr:

    def __init__(self, maxFramerate):
        self.maxFramerate = maxFramerate
        self.targetDelta = 1.0 / maxFramerate
        self.prevTime = time.clock()
        self.curTime = time.clock()
        self.delta = 0.0

    def Tick(self):
        self.curTime = time.clock()
        self.delta = self.curTime - self.prevTime
        self.prevTime = self.curTime

    def TimeSinceTick(self):
        return time.clock() - self.curTime

class AppData:

    def __init__(self):
        self.trainDataDirectory = None

    def GetTrainDirectory(self, event):
        self.trainDataDirectory = ChooseDirectory("Choose Training Directory")

class MainUI:

    def __init__(self):

        self.running = True

        self.tkRoot = Tk()
        self.tkRoot.geometry("640x480")

        self.mainContainer = Frame(self.tkRoot)
        self.mainContainer.pack()

        self.btnChooseTrainData = Button(self.mainContainer)
        self.btnChooseTrainData.configure(text="Choose training folder...")
        self.btnChooseTrainData.pack({"side": LEFT})

        self.btnLoadTraining = Button(self.mainContainer)
        self.btnLoadTraining.configure(text="Load training data")
        self.btnLoadTraining.pack({"side": LEFT})

        self.btnTrain = Button(self.mainContainer)
        self.btnTrain.configure({"text": "Train!"})
        self.btnTrain.pack({"side": LEFT})

        self.btnGenerate = Button(self.mainContainer)
        self.btnGenerate.configure(text="Generate new music!")
        self.btnGenerate.pack({"side": LEFT})

        self.btnQuit = Button(self.mainContainer)
        self.btnQuit.configure(text="Quit")
        self.btnQuit.pack({"side": RIGHT})
        self.btnQuit.bind("<ButtonRelease-1>", self.Quit)

    def Quit(self, event):
        self.running = False


#Use Tkinter to grab a directory choice from the user.
def ChooseDirectory(title=""):

    dialogOptions = {
        "title": title,
        "initialdir": ".",
        "mustexist": True
    }

    directory = filedialog.askdirectory(**dialogOptions)

    if directory == "":
        return None

    return directory
