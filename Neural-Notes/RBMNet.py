import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import glob
from tqdm import tqdm
import numpy as num
import os

MODEL_LOC = os.path.dirname(os.path.realpath(__file__)) + '/models/rbmnet.chkpt'
MODEL_LOC_CHECK = os.path.dirname(os.path.realpath(__file__)) + '/models/rbmnet.chkpt.index'
DEFAULT_TIMESTEPS = 15
DEFAULT_HNODES = 50
DEFAULT_EPOCHS = 200
DEFAULT_BATCHSIZE = 100
DEFAULT_LEARNRATE = 0.005

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

    def __init__(self, midiUtil, minTrainingSnippet=32):
        self.midi = midiUtil
        self.minTrainingSnippet = minTrainingSnippet

        self.trainDataset = []
        self.trained = False

        self.InitDefaultParameters()

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        return

    def InitDefaultParameters(self):
        self.notespan = self.midi.notespan
        self.timesteps = DEFAULT_TIMESTEPS

        self.vNodes = 2 * self.notespan * self.timesteps
        self.hNodes = DEFAULT_HNODES

        self.epochs = DEFAULT_EPOCHS
        self.batchSize = DEFAULT_BATCHSIZE
        self.learnRate = tf.constant(DEFAULT_LEARNRATE, tf.float32)

        self.notedata = tf.placeholder(tf.float32, [None, self.vNodes], name="notedata")
        self.wMatrix = tf.Variable(tf.random_normal([self.vNodes, self.hNodes], 0.01), name="wMatrix")

        self.vBias = tf.Variable(tf.zeros([1, self.vNodes], tf.float32, name="vBias"))
        self.hBias = tf.Variable(tf.zeros([1, self.hNodes], tf.float32, name="hBias"))

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

    def Train(self, event):

        if not len(self.trainDataset) > 0:
            print("Can't train without any data!")
            return

        with tf.Session() as session:

            init = tf.global_variables_initializer()
            session.run(init)

            netSaver = tf.train.Saver(tf.global_variables())

            for epoch in tqdm(range(self.epochs)):

                for tData in self.trainDataset:
                    tData = num.array(tData)
                    tData = tData[:int(num.floor(tData.shape[0] // self.timesteps) * self.timesteps)]
                    tData = num.reshape(tData, [tData.shape[0] // self.timesteps, tData.shape[1] * self.timesteps])

                    for i in range(1, len(tData), self.batchSize):
                        tr_x = tData[i:i + self.batchSize]
                        session.run(self.trainUpdate, feed_dict={self.notedata: tr_x})

            netSaver.save(session, MODEL_LOC)
            self.trained = True

        return

    def Generate(self, event):

        if not (self.trained or os.path.isfile(MODEL_LOC_CHECK)):
            print("Can't generate on an untrained network!")
            return

        with tf.Session() as session:

            netLoader = tf.train.Saver()
            netLoader.restore(session, MODEL_LOC)

            sample = Gibbs(k=1, x=self.notedata, wMatrix=self.wMatrix, hBias=self.hBias, vBias=self.vBias).eval(
                session=session, feed_dict={self.notedata: num.zeros((5, self.vNodes))})
            for i in range(sample.shape[0]):
                if not any(sample[i, :]):
                    continue
                S = num.reshape(sample[i, :], (self.timesteps, 2 * self.notespan))
                self.midi.FVtoMIDI(S, os.path.abspath(os.curdir) + "\\Test-Out-{}".format(i))

        return

