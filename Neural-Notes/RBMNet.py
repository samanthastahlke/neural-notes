import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import glob
from tqdm import tqdm
import numpy as num
import os
import shutil

MODEL_LOC = os.path.dirname(os.path.realpath(__file__)) + '/data/tmp_model/rbmnet.chkpt'
MODEL_LOC_CHECK = os.path.dirname(os.path.realpath(__file__)) + '/data/tmp_model/rbmnet.chkpt.index'
MODEL_SAVE_LOC = os.path.dirname(os.path.realpath(__file__)) + '/data/tmp_model'
SAMPLE_LOC = os.path.dirname(os.path.realpath(__file__)) + '/sampleout'

DEFAULT_TIMESTEPS = 48
DEFAULT_HNODES = 50
DEFAULT_EPOCHS = 75
DEFAULT_BATCHSIZE = 100
DEFAULT_LEARNRATE = 0.005
DEFAULT_SAMPLES = 5

def ProbSample(p):
    return tf.floor(p + tf.random_uniform(tf.shape(p), 0, 1))

def Gibbs(k, x, wMatrix, hBias, vBias):

    def GibbsStep(count, k, xk):
        hk = ProbSample(tf.sigmoid(tf.matmul(xk, wMatrix) + hBias))
        xk = ProbSample(tf.sigmoid(tf.matmul(hk, tf.transpose(wMatrix)) + vBias))
        return count + 1, k, xk

    ct = tf.constant(0)
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter, GibbsStep,
                                                   [ct, tf.constant(k), x])

    x_sample = tf.stop_gradient(x_sample)
    return x_sample

class RBMNet:

    def __init__(self, midiUtil):
        self.midi = midiUtil
        self.trainDataset = []
        self.InitNNParameters()

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        return

    def InitNNParameters(self):
        tf.reset_default_graph()

        self.notespan = self.midi.notespan
        self.timesteps = DEFAULT_TIMESTEPS
        self.tfTimesteps = tf.Variable(DEFAULT_TIMESTEPS, name="timesteps")
        self.genSample = DEFAULT_SAMPLES

        self.vNodes = 2 * self.notespan * self.timesteps
        self.hNodes = DEFAULT_HNODES

        self.epochs = DEFAULT_EPOCHS
        self.batchSize = DEFAULT_BATCHSIZE
        self.learnRate = tf.constant(DEFAULT_LEARNRATE, tf.float32)

        self.wMatrix = tf.Variable(tf.random_normal([self.vNodes, self.hNodes], 0.01), name="wMatrix")

        self.vBias = tf.Variable(tf.zeros([1, self.vNodes], tf.float32), name="vBias")
        self.hBias = tf.Variable(tf.zeros([1, self.hNodes], tf.float32), name="hBias")

        self.notedata = tf.placeholder(tf.float32, [None, self.vNodes])
        self.note_sample = Gibbs(k=1, x=self.notedata, wMatrix=self.wMatrix,
                                 hBias=self.hBias, vBias=self.vBias)

        self.hdata = ProbSample(tf.sigmoid(tf.matmul(self.notedata, self.wMatrix) + self.hBias))
        self.h_sample = ProbSample(tf.sigmoid(tf.matmul(self.note_sample, self.wMatrix) + self.hBias))

        self.elemShape = tf.cast(tf.shape(self.notedata)[0], tf.float32)
        self.wAdjust = tf.multiply(self.learnRate / self.elemShape,
                                   tf.subtract(tf.matmul(tf.transpose(self.notedata), self.hdata),
                                               tf.matmul(tf.transpose(self.note_sample), self.h_sample)))

        self.vBAdjust = tf.multiply(self.learnRate / self.elemShape,
                                    tf.reduce_sum(tf.subtract(self.notedata, self.note_sample), 0, True))
        self.hBAdjust = tf.multiply(self.learnRate / self.elemShape,
                                    tf.reduce_sum(tf.subtract(self.hdata, self.h_sample), 0, True))

        self.trainUpdate = [self.wMatrix.assign_add(self.wAdjust),
                            self.vBias.assign_add(self.vBAdjust),
                            self.hBias.assign_add(self.hBAdjust)]

    def LoadTrainingSet(self, directory):

        self.trainDataset = []

        if directory is None:
            print("Can't load training data - no valid directory specified!")
            return False

        fileset = glob.glob("{}/*.mid*".format(directory))

        for midifile in tqdm(fileset):
            try:
                fv = num.array(self.midi.MIDItoFV(midifile))

                if num.array(fv).shape[0] > self.timesteps:
                    self.trainDataset.append(fv)
            except Exception as e:
                print(e)

        print("Loaded {} MIDI files for training.".format(len(self.trainDataset)))

        return True

    def Train(self, event, saveDir):

        if not len(self.trainDataset) > 0:
            print("Can't train without any data!")
            return False

        modelSaveLoc = MODEL_SAVE_LOC
        saveTmp = False

        if saveDir is not None and saveDir and os.path.isdir(saveDir) and os.listdir(saveDir) == []:
            modelSaveLoc = saveDir
            os.rmdir(saveDir)
            saveTmp = True
            print("Model will be saved to " + modelSaveLoc)

        shutil.rmtree(MODEL_SAVE_LOC)
        netBuilder = tf.saved_model.builder.SavedModelBuilder(modelSaveLoc)

        if saveTmp:
            tmpNetBuilder = tf.saved_model.builder.SavedModelBuilder(MODEL_SAVE_LOC)

        with tf.Session() as session:

            init = tf.global_variables_initializer()
            session.run(init)

            for epoch in tqdm(range(self.epochs)):

                for tData in self.trainDataset:
                    tData = num.array(tData)
                    tData = tData[:int(num.floor(tData.shape[0] // self.timesteps) * self.timesteps)]
                    tData = num.reshape(tData, [tData.shape[0] // self.timesteps, tData.shape[1] * self.timesteps])

                    for i in range(1, len(tData), self.batchSize):
                        tr_x = tData[i:i + self.batchSize]
                        session.run(self.trainUpdate, feed_dict={self.notedata: tr_x})

            netBuilder.add_meta_graph_and_variables(session, ["RBMNet"])

            if saveTmp:
                tmpNetBuilder.add_meta_graph_and_variables(session, ["RBMNet"])

        netBuilder.save()

        if saveTmp:
            tmpNetBuilder.save()

        print("Model stored in " + modelSaveLoc)
        return True

    def IsTmpModelStored(self):
        modelLoadCheck = MODEL_SAVE_LOC + "/saved_model.pb"
        return os.path.isfile(modelLoadCheck)

    def Generate(self, event, loadDir, saveDir):

        sampleSaveLoc = SAMPLE_LOC

        if saveDir is not None and saveDir and os.path.isdir(saveDir):
            sampleSaveLoc = saveDir

        modelLoadLoc = MODEL_SAVE_LOC

        if loadDir is not None and loadDir and os.path.isdir(loadDir):
            modelLoadLoc = loadDir

        modelLoadCheck = modelLoadLoc + "/saved_model.pb"

        if not os.path.isfile(modelLoadCheck):
            print("Can't generate with no network available!")
            return False

        print("Loading model from " + modelLoadLoc + "...")
        tf.reset_default_graph()

        with tf.Session() as session:

            session.run(tf.global_variables_initializer())
            tf.saved_model.loader.load(session, ["RBMNet"], modelLoadLoc)

            for v in tf.global_variables():
                if "wMatrix" in v.name:
                    self.wMatrix = v
                elif "vBias" in v.name:
                    self.vBias = v
                elif "hBias" in v.name:
                    self.hBias = v
                elif "timesteps" in v.name:
                    self.tfTimesteps = v

            self.timesteps = session.run(self.tfTimesteps)
            self.vNodes = 2 * self.notespan * self.timesteps

            self.notedata = tf.placeholder(tf.float32, [None, self.vNodes])

            sample = Gibbs(k=1, x=self.notedata, wMatrix=self.wMatrix, hBias=self.hBias, vBias=self.vBias).eval(
                session=session, feed_dict={self.notedata: num.zeros((self.genSample, self.vNodes))})
            for i in range(sample.shape[0]):
                if not any(sample[i, :]):
                    continue
                S = num.reshape(sample[i, :], (self.timesteps, 2 * self.notespan))
                self.midi.FVtoMIDI(S, sampleSaveLoc + "/OutputSample-{}".format(i))

        print("Saved samples to " + sampleSaveLoc)

        return True