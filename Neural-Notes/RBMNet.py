import tensorflow as tf
import glob
import numpy as num
from tqdm import tqdm
import os

class RBMNet:

    def __init__(self, midiUtil, minTrainingSnippet=32):
        self.midi = midiUtil
        self.minTrainingSnippet = minTrainingSnippet
        self.trainDataset = []
        return

    def LoadTrainingSet(self, directory):

        if directory is None:
            print("Can't load training data - no valid directory specified!")
            return

        self.trainDataset = []
        fileset = glob.glob("{}/*.mid*".format(directory))

        for midifile in fileset:
            try:
                fv = num.array(self.midi.MIDItoFV(midifile))

                if num.array(fv).shape[0] > self.minTrainingSnippet:
                    self.trainDataset.append(fv)
            except Exception as e:
                print(e)

        print("Loaded {} MIDI files for training.".format(len(self.trainDataset)))

        if len(self.trainDataset) > 0:
            self.midi.FVtoMIDI(self.trainDataset[0], os.path.abspath(os.curdir) + "\\Test-Converted")

        return

