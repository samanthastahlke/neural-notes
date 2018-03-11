import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import glob
from tqdm import tqdm
import numpy as num
import os

class RBMNet:

    def __init__(self, midiUtil, minTrainingSnippet=32):
        self.midi = midiUtil
        self.minTrainingSnippet = minTrainingSnippet
        self.trainDataset = []
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

    def Train(self, event):

        if not len(self.trainDataset) > 0:
            print("Can't train without any data!")
            return

        notespan = self.midi.notespan

        timesteps = 15
        vLayerNodes = 2 * notespan * timesteps
        hLayerNodes = 50

        epochCount = 200
        batchSize = 100

        learningRate = tf.constant(0.005, tf.float32)

        x = tf.placeholder(tf.float32, [None, vLayerNodes], name="x")
        weightMatrix = tf.Variable(tf.random_normal([vLayerNodes, hLayerNodes], 0.01), name="weightMatrix")

        hBias = tf.Variable(tf.zeros([1, hLayerNodes], tf.float32, name="hBias"))
        vBias = tf.Variable(tf.zeros([1, vLayerNodes], tf.float32, name="vBias"))

        def ProbSample(p):
            return tf.floor(p + tf.random_uniform(tf.shape(p), 0, 1))

        def Gibbs(k):
            def GibbsStep(count, k, xk):
                hk = ProbSample(tf.sigmoid(tf.matmul(xk, weightMatrix) + hBias))
                xk = ProbSample(tf.sigmoid(tf.matmul(hk, tf.transpose(weightMatrix)) + vBias))
                return count + 1, k, xk

            ct = tf.constant(0)
            [_,_,x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter, GibbsStep, [ct, tf.constant(k), x])

            x_sample = tf.stop_gradient(x_sample)
            return x_sample

        x_sample = Gibbs(1)
        h = ProbSample(tf.sigmoid(tf.matmul(x, weightMatrix) + hBias))
        h_sample = ProbSample(tf.sigmoid(tf.matmul(x_sample, weightMatrix) + hBias))

        size_bt = tf.cast(tf.shape(x)[0], tf.float32)
        W_adder = tf.multiply(learningRate / size_bt,
                              tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))

        bv_adder = tf.multiply(learningRate / size_bt,
                               tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
        bh_adder = tf.multiply(learningRate / size_bt,
                               tf.reduce_sum(tf.subtract(h, h_sample), 0, True))

        trainUpdate = [weightMatrix.assign_add(W_adder), vBias.assign_add(bv_adder), hBias.assign_add(bh_adder)]

        with tf.Session() as session:

            init = tf.global_variables_initializer()
            session.run(init)

            for epoch in tqdm(range(epochCount)):

                for tData in self.trainDataset:
                    tData = num.array(tData)
                    tData = tData[:int(num.floor(tData.shape[0] // timesteps) * timesteps)]
                    tData = num.reshape(tData, [tData.shape[0] // timesteps, tData.shape[1] * timesteps])

                    for i in range(1, len(tData), batchSize):
                        tr_x = tData[i:i + batchSize]
                        session.run(trainUpdate, feed_dict={x: tr_x})


            sample = Gibbs(1).eval(session=session, feed_dict={x: num.zeros((5, vLayerNodes))})
            for i in range(sample.shape[0]):
                if not any(sample[i, :]):
                    continue
                S = num.reshape(sample[i, :], (timesteps, 2 * notespan))
                self.midi.FVtoMIDI(S, os.path.abspath(os.curdir) + "\\Test-Out-{}".format(i))

        return

    def Generate(self):
        return

