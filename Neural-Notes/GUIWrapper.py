import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time

BG_COL = '#000000'
TXT_COL = '#ffffff'
BTN_COL = '#202020'
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

def configUIButtonSquare(btn):
    btn.configure(bg=BTN_COL, fg=TXT_COL,
                  activebackground=FIELD_COL,
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
    tog.configure(bg=BG_COL, fg=TXT_COL,
                  activebackground=BG_COL,
                  activeforeground=TXT_COL,
                  selectcol=BG_COL,
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
        self.tkRoot.title("Neural Notes")
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

        self.btnTrain = tk.Button(self.mainContainer,
                                  text="Train...")
        configUIButton(self.btnTrain)
        self.btnTrain.pack(pady=PADDING)
        self.btnTrain.bind("<ButtonRelease-1>", self.GoTrain)

        self.btnLoadModel = tk.Button(self.mainContainer,
                                      text="Load Model...")
        configUIButton(self.btnLoadModel)
        self.btnLoadModel.pack(pady=PADDING)

        self.btnSaveModel = tk.Button(self.mainContainer,
                                      text="Save Current Model...")
        configUIButton(self.btnSaveModel)
        self.btnSaveModel.pack(pady=PADDING)

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

        row = 0
        col = 0

        self.tLblTitle = tk.Label(self.trainContainer,
                                  text="Training")
        configUILabel(self.tLblTitle)
        self.tLblTitle.configure(foreground=LIT_COL)
        self.tLblTitle.grid(row=row,column=col,sticky=tk.W,pady=PADDING*2,padx=PADDING*2)

        row += 1

        self.tLblStatus = tk.Label(self.trainContainer,
                                text="STATUS: ---")
        configUILabel(self.tLblStatus)
        self.tLblStatus.grid(row=row,column=col,sticky=tk.W,padx=PADDING*2)

        row += 1
        col = 0

        self.tBtnChooseData = tk.Button(self.trainContainer,
                                        text="Choose training folder...")
        configUIButtonSquare(self.tBtnChooseData)
        self.tBtnChooseData.grid(row=row,column=col,sticky=tk.W,pady=PADDING*2,padx=PADDING*2)

        row += 1

        #Training fields.
        self.trainContainer.grid_rowconfigure(row,weight=1)
        self.trainFieldsContainer = tk.Frame(self.trainContainer, background=BG_COL)
        self.trainFieldsContainer.grid(row=row,column=0,columnspan=2,sticky=tk.W,pady=PADDING*4,padx=PADDING*2)

        fRow = 0

        self.tLblEpochs = tk.Label(self.trainFieldsContainer,
                                   text="Epochs: ")
        configUILabel(self.tLblEpochs)
        self.tLblEpochs.grid(row=fRow,column=0,sticky=tk.E,pady=PADDING*2,padx=PADDING)
        self.tTxtEpochs = tk.Entry(self.trainFieldsContainer)
        configUIField(self.tTxtEpochs)
        self.tTxtEpochs.grid(row=fRow,column=1,sticky=tk.W,pady=PADDING*2,padx=PADDING)
        self.tTxtEpochs.insert(0, "---")

        fRow += 1

        self.tLblLearn = tk.Label(self.trainFieldsContainer,
                                  text="Learning Rate: ")
        configUILabel(self.tLblLearn)
        self.tLblLearn.grid(row=fRow,column=0,sticky=tk.E,pady=PADDING*2,padx=PADDING)

        self.tTxtLearn = tk.Entry(self.trainFieldsContainer)
        configUIField(self.tTxtLearn)
        self.tTxtLearn.grid(row=fRow,column=1,sticky=tk.W,pady=PADDING*2,padx=PADDING)
        self.tTxtLearn.insert(0, "---")

        fRow += 1

        self.tLblNodes = tk.Label(self.trainFieldsContainer,
                                  text="Hidden Nodes: ")
        configUILabel(self.tLblNodes)
        self.tLblNodes.grid(row=fRow,column=0,sticky=tk.E,pady=PADDING*2,padx=PADDING)

        self.tTxtNodes = tk.Entry(self.trainFieldsContainer)
        configUIField(self.tTxtNodes)
        self.tTxtNodes.grid(row=fRow,column=1,sticky=tk.W,pady=PADDING*2,padx=PADDING)
        self.tTxtNodes.insert(0, "---")

        fRow += 1

        self.tLblTimesteps = tk.Label(self.trainFieldsContainer,
                                      text="Timesteps: ")
        configUILabel(self.tLblTimesteps)
        self.tLblTimesteps.grid(row=fRow,column=0,sticky=tk.E,pady=PADDING*2,padx=PADDING)

        self.tTxtTimesteps = tk.Entry(self.trainFieldsContainer)
        configUIField(self.tTxtTimesteps)
        self.tTxtTimesteps.grid(row=fRow,column=1,sticky=tk.W,pady=PADDING*2,padx=PADDING)
        self.tTxtTimesteps.insert(0, "---")

        row += 1
        col = 0

        self.trainContainer.grid_rowconfigure(row,weight=1)
        self.tBtnChooseSave = tk.Button(self.trainContainer,
                                        text="Choose model save directory...")
        configUIButtonSquare(self.tBtnChooseSave)
        self.tBtnChooseSave.grid(row=row,column=col,sticky="sw",pady=PADDING,padx=PADDING*2)

        row += 1
        col = 0

        self.saveModel = False
        self.tTogSave = tk.Checkbutton(self.trainContainer,
                                       var=self.saveModel,
                                       text="Save model to ")
        configUIToggle(self.tTogSave)
        self.tTogSave.grid(row=row,column=col,sticky=tk.W,pady=PADDING,padx=PADDING)

        col += 1

        self.tLblSave = tk.Label(self.trainContainer,
                                 text="---",
                                 wraplength=400,
                                 justify=tk.LEFT)
        configUILabel(self.tLblSave)
        self.tLblSave.grid(row=row,column=col,sticky=tk.W,pady=PADDING,padx=PADDING)

        row += 1
        col = 0

        self.tBtnBack = tk.Button(self.trainContainer,
                                  text="Back")
        configUIButtonSquare(self.tBtnBack)
        self.tBtnBack.grid(row=row,column=col,sticky="sw",pady=PADDING*2,padx=PADDING*2)
        self.tBtnBack.bind("<ButtonRelease-1>", self.GoMain)

        col += 2
        self.trainContainer.grid_columnconfigure(col,weight=1)
        self.tBtnLoadData = tk.Button(self.trainContainer,
                                      text="Load Data")
        configUIButtonSquare(self.tBtnLoadData)
        self.tBtnLoadData.grid(row=row,column=col,sticky="se",pady=PADDING*2)

        col += 1

        self.tBtnTrain = tk.Button(self.trainContainer,
                                   text="Train!")
        configUIButtonSquare(self.tBtnTrain)
        self.tBtnTrain.grid(row=row,column=col,sticky="se",pady=PADDING*2,padx=PADDING*2)

        #Generation screen.
        self.genContainer = tk.Frame(self.rootContainer, background=BG_COL)
        self.genContainer.grid_configure(row=0, column=0, sticky="nsew")

        row = 0
        col = 0

        self.gLblTitle = tk.Label(self.genContainer,
                                  text="Generation")
        configUILabel(self.gLblTitle)
        self.gLblTitle.grid(row=row,column=col,sticky=tk.W,pady=PADDING*2,padx=PADDING*2)

        row += 1

        self.gLblStatus = tk.Label(self.genContainer,
                                   text="STATUS: ---")
        configUILabel(self.gLblStatus)
        self.gLblStatus.grid(row=row,column=col,sticky=tk.W,padx=PADDING*2)

        row += 1

        self.gBtnChooseModel = tk.Button(self.genContainer,
                                        text="Load Model...")
        configUIButtonSquare(self.gBtnChooseModel)
        self.gBtnChooseModel.grid(row=row,column=col,sticky=tk.W,pady=PADDING,padx=PADDING*2)

        row += 1

        self.genContainer.rowconfigure(row,weight=1)
        self.genFieldsContainer = tk.Frame(self.genContainer,background=BG_COL)
        self.genFieldsContainer.grid_configure(row=row,column=col,sticky=tk.W,pady=PADDING*2,padx=PADDING*2)

        fRow = 0

        self.gLblSamples = tk.Label(self.genFieldsContainer,
                                   text="Samples: ")
        configUILabel(self.gLblSamples)
        self.gLblSamples.grid(row=fRow,column=0,sticky=tk.E,pady=PADDING*2,padx=PADDING)

        self.gTxtSamples = tk.Entry(self.genFieldsContainer)
        configUIField(self.gTxtSamples)
        self.gTxtSamples.grid(row=fRow,column=1,sticky=tk.W,pady=PADDING*2,padx=PADDING)
        self.gTxtSamples.insert(0, "---")

        fRow += 1

        self.gLblTimescale = tk.Label(self.genFieldsContainer,
                                      text="MIDI Tick Scale: ")
        configUILabel(self.gLblTimescale)
        self.gLblTimescale.grid(row=fRow,column=0,sticky=tk.E,pady=PADDING*2,padx=PADDING)

        self.gTxtTimescale = tk.Entry(self.genFieldsContainer)
        configUIField(self.gTxtTimescale)
        self.gTxtTimescale.grid(row=fRow,column=1,sticky=tk.W,pady=PADDING*2,padx=PADDING)
        self.gTxtTimescale.insert(0, "---")

        row += 1

        self.genContainer.rowconfigure(row,weight=1)
        self.gBtnChooseSave = tk.Button(self.genContainer,
                                        text="Choose sample save directory...")
        configUIButtonSquare(self.gBtnChooseSave)
        self.gBtnChooseSave.grid(row=row,column=col,sticky="sw",pady=PADDING,padx=PADDING*2)

        row += 1

        self.gLblSaveDir = tk.Label(self.genContainer,
                                    text="Saving samples to ---",
                                    wraplength=400,
                                    justify=tk.LEFT)
        configUILabel(self.gLblSaveDir)
        self.gLblSaveDir.grid(row=row,column=col,sticky="sw",pady=PADDING,padx=PADDING*2)

        row += 1

        self.gBtnBack = tk.Button(self.genContainer,
                                  text="Back")
        configUIButtonSquare(self.gBtnBack)
        self.gBtnBack.grid(row=row,column=col,sticky="sw",pady=PADDING*2,padx=PADDING*2)
        self.gBtnBack.bind("<ButtonRelease-1>", self.GoMain)

        col += 2
        self.genContainer.columnconfigure(col,weight=1)

        self.gBtnGen = tk.Button(self.genContainer,
                                 text="Generate!")
        configUIButtonSquare(self.gBtnGen)
        self.gBtnGen.grid(row=row,column=col,sticky="se",pady=PADDING*2,padx=PADDING*2)

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
