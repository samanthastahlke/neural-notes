import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time

BG_COL = '#000000'
TXT_COL = '#ffffff'
LIT_COL = '#00ff96'
PADDING = 4

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

def configUIButton(btn):
    btn.configure(bg=BG_COL, fg=TXT_COL,
                  activebackground=BG_COL,
                  activeforeground=LIT_COL,
                  relief=tk.FLAT)
    btn.bind('<Enter>', lambda event, b=btn:
                btn.configure(state='active'))

class MainUI:

    def __init__(self):

        self.running = True

        self.tkRoot = tk.Tk()
        self.tkRoot.geometry("640x480")
        self.tkRoot.configure(bg=BG_COL,
                              highlightbackground=BG_COL,
                              highlightthickness=0)

        iTitle = Image.open('img/title.png')
        szTitle = 512, 128
        iTitle.thumbnail(szTitle, Image.ANTIALIAS)
        self.imgTitle = ImageTk.PhotoImage(iTitle)
        self.dispTitle = tk.Label(self.tkRoot, image=self.imgTitle, bg='#000000')
        self.dispTitle.pack(pady=(100,10))

        self.mainContainer = tk.Frame(self.tkRoot,background=BG_COL)
        self.mainContainer.pack()

        self.btnChooseTrainData = tk.Button(self.mainContainer,
                                            text="Choose training folder...")
        configUIButton(self.btnChooseTrainData)
        self.btnChooseTrainData.pack(pady=PADDING)

        self.btnLoadTraining = tk.Button(self.mainContainer,
                                         text="Load training data")
        configUIButton(self.btnLoadTraining)
        self.btnLoadTraining.pack(pady=PADDING)

        self.btnTrain = tk.Button(self.mainContainer,
                                  text="Train!")
        configUIButton(self.btnTrain)
        self.btnTrain.pack(pady=PADDING)

        self.btnGenerate = tk.Button(self.mainContainer,
                                     text="Generate new music!")
        configUIButton(self.btnGenerate)
        self.btnGenerate.pack(pady=PADDING)

        self.btnQuit = tk.Button(self.mainContainer,
                                 text="Quit")
        configUIButton(self.btnQuit)
        self.btnQuit.pack(pady=PADDING)
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
