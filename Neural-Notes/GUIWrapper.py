import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time

BG_COL = '#000000'
TXT_COL = '#ffffff'
FIELD_COL = '#303030'
LIT_COL = '#00ff96'
PADDING = 4

'''
FrameMgr class.
Lightweight frame timer used to structure our main update loop.
'''
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

'''
AppData class. 
Small container for storing needed data that isn't tied directly to the RBM.
'''
class AppData:

    def __init__(self):
        self.trainDataDirectory = None
        self.modelLoadDirectory = None
        self.modelSaveDirectory = None
        self.sampleSaveDirectory = None

    def GetTrainDirectory(self, event):
        self.trainDataDirectory = ChooseDirectory("Choose training data folder...")

    def GetModelLoadDirectory(self, event):
        self.modelLoadDirectory = ChooseDirectory("Select a folder containing a trained model...")

    def GetModelSaveDirectory(self, event):
        self.modelSaveDirectory = filedialog.asksaveasfile(mode='w', defaultextension="")

    def GetSampleSaveDirectory(self, event):
        self.sampleSaveDirectory = ChooseDirectory("Select an output folder...")

'''
Utility functions for configuring Tkinter UI elements according to our colour scheme/layout.
Extracted to reduce duplication.
'''
def configUIButton(btn):
    btn.configure(bg=BG_COL, fg=TXT_COL,
                  activebackground=BG_COL,
                  activeforeground=LIT_COL,
                  relief=tk.FLAT)
    btn.bind('<Enter>', lambda event, b=btn:
                btn.configure(state='active'))

def configUILabel(lbl):
    lbl.configure(bg=BG_COL, fg=TXT_COL)

def configUIField(txt):
    txt.configure(bg=FIELD_COL, fg=TXT_COL,
                  insertbackground=TXT_COL,
                  relief=tk.FLAT)

def configUIToggle(tog):
    tog.configure(bg=BG_COL, fg=FIELD_COL,
                  activebackground=BG_COL,
                  relief=tk.FLAT)
'''
MainUI class. 
Responsible for initializing and storing our (hardcoded) Tkinter UI.
All UI layout/initialization happens in __init__.
Callbacks relying on other functionality are initialized in NeuralNotes.py.
'''
class MainUI:

    def __init__(self):

        self.running = True

        # Initialize Tkinter.
        self.tkRoot = tk.Tk()
        self.tkRoot.geometry("640x480")
        self.tkRoot.configure(bg=BG_COL,
                              highlightbackground=BG_COL,
                              highlightthickness=0)

        #Create our root container.
        self.rootContainer = tk.Frame(self.tkRoot)
        self.rootContainer.pack(side="top", fill="both", expand=True)
        self.rootContainer.grid_rowconfigure(0, weight=1)
        self.rootContainer.grid_columnconfigure(0, weight=1)

        #Main screen.
        self.mainContainer = tk.Frame(self.rootContainer,background=BG_COL)
        self.mainContainer.grid(row=0, column=0, sticky="nsew")

        #Title image. Because we're classy.
        iTitle = Image.open('data/img/title.png')
        szTitle = 512, 128
        iTitle.thumbnail(szTitle, Image.ANTIALIAS)
        self.imgTitle = ImageTk.PhotoImage(iTitle)
        self.dispTitle = tk.Label(self.mainContainer, image=self.imgTitle, bg='#000000')
        self.dispTitle.pack(pady=(100,10))

        self.btnChooseTrainData = tk.Button(self.mainContainer,
                                            text="Choose training folder...")
        configUIButton(self.btnChooseTrainData)
        self.btnChooseTrainData.pack(pady=PADDING)

        self.btnLoadTraining = tk.Button(self.mainContainer,
                                         text="Load training data")
        configUIButton(self.btnLoadTraining)
        self.btnLoadTraining.pack(pady=PADDING)

        self.btnTrain = tk.Button(self.mainContainer,
                                  text="Train...")
        configUIButton(self.btnTrain)
        self.btnTrain.pack(pady=PADDING)
        self.btnTrain.bind("<ButtonRelease-1>", self.GoTrain)

        self.btnGenerate = tk.Button(self.mainContainer,
                                     text="Generate...")
        configUIButton(self.btnGenerate)
        self.btnGenerate.pack(pady=PADDING)
        self.btnGenerate.bind("<ButtonRelease-1>", self.GoGen)

        self.btnQuit = tk.Button(self.mainContainer,
                                 text="Quit")
        configUIButton(self.btnQuit)
        self.btnQuit.pack(pady=PADDING)
        self.btnQuit.bind("<ButtonRelease-1>", self.Quit)

        #Training screen.
        self.trainContainer = tk.Frame(self.rootContainer, background=BG_COL)
        self.trainContainer.grid_configure(row=0, column=0, sticky="nsew")

        self.tLblTitle = tk.Label(self.trainContainer,
                                  text="Training")
        configUILabel(self.tLblTitle)
        self.tLblTitle.pack(pady=PADDING)

        self.tLblStatus = tk.Label(self.trainContainer,
                                text="STATUS: ")
        configUILabel(self.tLblStatus)
        self.tLblStatus.pack(pady=PADDING)

        self.tBtnChooseData = tk.Button(self.trainContainer,
                                        text="Choose training folder...")
        configUIButton(self.tBtnChooseData)
        self.tBtnChooseData.pack(pady=PADDING)

        self.tLblEpochs = tk.Label(self.trainContainer,
                                   text="Epochs: ")
        configUILabel(self.tLblEpochs)
        self.tLblEpochs.pack(pady=PADDING)

        self.tTxtEpochs = tk.Entry(self.trainContainer)
        configUIField(self.tTxtEpochs)
        self.tTxtEpochs.pack(pady=PADDING)
        self.tTxtEpochs.insert(0, "---")

        self.tLblLearn = tk.Label(self.trainContainer,
                                  text="Learning Rate: ")
        configUILabel(self.tLblLearn)
        self.tLblLearn.pack(pady=PADDING)

        self.tTxtLearn = tk.Entry(self.trainContainer)
        configUIField(self.tTxtLearn)
        self.tTxtLearn.pack(pady=PADDING)
        self.tTxtLearn.insert(0, "---")

        self.tLblNodes = tk.Label(self.trainContainer,
                                  text="Hidden Nodes: ")
        configUILabel(self.tLblNodes)
        self.tLblNodes.pack(pady=PADDING)

        self.tTxtNodes = tk.Entry(self.trainContainer)
        configUIField(self.tTxtNodes)
        self.tTxtNodes.pack(pady=PADDING)
        self.tTxtNodes.insert(0, "---")

        self.tLblTimesteps = tk.Label(self.trainContainer,
                                      text="Timesteps: ")
        configUILabel(self.tLblTimesteps)
        self.tLblTimesteps.pack(pady=PADDING)

        self.tTxtTimesteps = tk.Entry(self.trainContainer)
        configUIField(self.tTxtTimesteps)
        self.tTxtTimesteps.pack(pady=PADDING)
        self.tTxtTimesteps.insert(0, "---")

        self.tBtnChooseSave = tk.Button(self.trainContainer,
                                        text="Choose model save directory...")
        configUIButton(self.tBtnChooseSave)
        self.tBtnChooseSave.pack(pady=PADDING)

        self.saveModel = False
        self.tTogSave = tk.Checkbutton(self.trainContainer,
                                       var=self.saveModel)
        configUIToggle(self.tTogSave)
        self.tTogSave.pack(pady=PADDING)

        self.tLblSave = tk.Label(self.trainContainer,
                                 text="Save model to ---")
        configUILabel(self.tLblSave)
        self.tLblSave.pack(pady=PADDING)


        self.tBtnBack = tk.Button(self.trainContainer,
                                  text="Back")
        configUIButton(self.tBtnBack)
        self.tBtnBack.pack(pady=PADDING)
        self.tBtnBack.bind("<ButtonRelease-1>", self.GoMain)

        self.tBtnTrain = tk.Button(self.trainContainer,
                                   text="Train!")
        configUIButton(self.tBtnTrain)
        self.tBtnTrain.pack(pady=PADDING)

        #Generation screen.
        self.genContainer = tk.Frame(self.rootContainer, background=BG_COL)
        self.genContainer.grid_configure(row=0, column=0, sticky="nsew")

        self.gLblTitle = tk.Label(self.genContainer,
                                  text="Generation")
        configUILabel(self.gLblTitle)
        self.gLblTitle.pack(pady=PADDING)

        self.gLblStatus = tk.Label(self.genContainer,
                                   text="STATUS: ")
        configUILabel(self.gLblStatus)
        self.gLblStatus.pack(pady=PADDING)

        self.gBtnChooseModel = tk.Button(self.genContainer,
                                        text="Load Model...")
        configUIButton(self.gBtnChooseModel)
        self.gBtnChooseModel.pack(pady=PADDING)

        self.gLblSamples = tk.Label(self.genContainer,
                                   text="Samples: ")
        configUILabel(self.gLblSamples)
        self.gLblSamples.pack(pady=PADDING)

        self.gTxtSamples = tk.Entry(self.genContainer)
        configUIField(self.gTxtSamples)
        self.gTxtSamples.pack(pady=PADDING)
        self.gTxtSamples.insert(0, "---")

        self.gBtnChooseSave = tk.Button(self.genContainer,
                                        text="Choose sample save directory...")
        configUIButton(self.gBtnChooseSave)
        self.gBtnChooseSave.pack(pady=PADDING)

        self.gBtnBack = tk.Button(self.genContainer,
                                  text="Back")
        configUIButton(self.gBtnBack)
        self.gBtnBack.pack(pady=PADDING)
        self.gBtnBack.bind("<ButtonRelease-1>", self.GoMain)

        self.gBtnGen = tk.Button(self.genContainer,
                                 text="Generate!")
        configUIButton(self.gBtnGen)
        self.gBtnGen.pack(pady=PADDING)

        self.mainContainer.tkraise()

    def GoMain(self, event):
        self.mainContainer.tkraise()

    def GoTrain(self, event):
        self.trainContainer.tkraise()

    def GoGen(self, event):
        self.genContainer.tkraise()

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
