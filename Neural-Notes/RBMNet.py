'''
RBMNET.py

This script manages training of the neural net.

DEPENDENCIES:

Tensorflow - machine learning library.
Glob and Shutil - directory utilities.
Tqdm - console progress bars.
Numpy - math library.
OS - directory path/tensorflow logging.
'''

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import glob
import shutil
from tqdm import tqdm
import numpy as num
import os

#Default paths for model/sample saving.
MODEL_SAVE_LOC = os.path.dirname(os.path.realpath(__file__)) + '/data/tmp_model'
SAMPLE_LOC = os.path.dirname(os.path.realpath(__file__)) + '/sampleout'

#Starting values for neural net parameters.
#Yes, some of these are changed by the UI.
#But if Python can't adhere to a bloody constant standard, why should I?
DEFAULT_TIMESTEPS = 48
DEFAULT_HNODES = 50
DEFAULT_EPOCHS = 75
DEFAULT_BATCHSIZE = 100
DEFAULT_LEARNRATE = 0.005
DEFAULT_SAMPLES = 5

#Probabilistic random tensor sampling.
def ProbSample(p):
    return tf.floor(p + tf.random_uniform(tf.shape(p), 0, 1))

#"Gibbs Sampling" - the method for sampling from an RBM.
#Used to generate our sample.
def Gibbs(k, x, wMatrix, hBias, vBias):

    def GibbsStep(count, k, xk):
        #Propagates visible layer (initially equal to xk) forward, getting a sample of the hidden layer.
        hk = ProbSample(tf.sigmoid(tf.matmul(xk, wMatrix) + hBias))
        #Propagates hidden sample backwards, reconstructing the visible layer.
        xk = ProbSample(tf.sigmoid(tf.matmul(hk, tf.transpose(wMatrix)) + vBias))
        return count + 1, k, xk

    #Use Tensorflow's while loop to run k Gibbs iterations.
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

        #Parameters constraining qualities of the training/generated samples.
        #Span of notes/MIDI timesteps to consider (affects both training/generation), and number of samples to generate.
        #MIDI timesteps and number of samples are customizable from UI.
        #These parameters also control the size of our visible layer.
        self.notespan = self.midi.notespan
        self.timesteps = DEFAULT_TIMESTEPS
        self.tfTimesteps = tf.Variable(DEFAULT_TIMESTEPS, name="timesteps")
        self.genSample = DEFAULT_SAMPLES

        #Size of our hidden and visible layers.
        self.vNodes = 2 * self.notespan * self.timesteps
        self.hNodes = DEFAULT_HNODES

        #Parameters controlling training.
        #Epochs/learn rate are customizable from the UI.
        self.epochs = DEFAULT_EPOCHS
        self.batchSize = DEFAULT_BATCHSIZE
        self.learnRate = tf.constant(DEFAULT_LEARNRATE, tf.float32)

        #Initialize our weight matrix and bias vectors.
        #Weight matrix starts as random values, biases start as zeroes.
        #(This way, an untrained network will output a very noisy mess instead of silence.
        # This is very useful for debugging as it helps us differentiate between untrained and overtrained networks,
        # since overtrained networks tend to output silence, which untrained networks will output what sounds like
        # approximately half a dozen toddlers jumping up and down on the same piano.)
        self.wMatrix = tf.Variable(tf.random_normal([self.vNodes, self.hNodes], 0.01), name="wMatrix")

        self.vBias = tf.Variable(tf.zeros([1, self.vNodes], tf.float32), name="vBias")
        self.hBias = tf.Variable(tf.zeros([1, self.hNodes], tf.float32), name="hBias")

        #Note that everything from here on down is essentially "placeholder" - Tensorflow does some magic with
        #references and function pointers that will allow these values to continuously update once we trigger
        #the training routine defined below.

        #Tensorflow "placeholder" variable - this is our "feature vector" and will be fed with training data.
        self.notedata = tf.placeholder(tf.float32, [None, self.vNodes])
        #This variable will be used to sample from our network while it is training.
        self.note_sample = Gibbs(k=1, x=self.notedata, wMatrix=self.wMatrix,
                                 hBias=self.hBias, vBias=self.vBias)

        #Hidden layer placeholder data/sample.
        self.hdata = ProbSample(tf.sigmoid(tf.matmul(self.notedata, self.wMatrix) + self.hBias))
        self.h_sample = ProbSample(tf.sigmoid(tf.matmul(self.note_sample, self.wMatrix) + self.hBias))

        #Used for Tensorflow to keep track of the shape (dimensionality) of the network.
        self.elemShape = tf.cast(tf.shape(self.notedata)[0], tf.float32)

        #This part of our training routine will adjust the matrix weights.
        #Note the use of tf.subtract, which is essentially our cost function.
        self.wAdjust = tf.multiply(self.learnRate / self.elemShape,
                                   tf.subtract(tf.matmul(tf.transpose(self.notedata), self.hdata),
                                               tf.matmul(tf.transpose(self.note_sample), self.h_sample)))

        #The following two operations adjust the biases in a similar fashion to above.
        #In essence, our "cost function" is attempting to minimize the difference between the data given to the
        #network (the actual visible layer) and the data reconstructed by the network (the estimate of the visible
        #layer obtained via Gibbs sampling).
        self.vBAdjust = tf.multiply(self.learnRate / self.elemShape,
                                    tf.reduce_sum(tf.subtract(self.notedata, self.note_sample), 0, True))
        self.hBAdjust = tf.multiply(self.learnRate / self.elemShape,
                                    tf.reduce_sum(tf.subtract(self.hdata, self.h_sample), 0, True))

        #This defines a training update routine in Tensorflow.
        #Adjust weights and biases according to calculated "nudges".
        #We will trigger this repeatedly during the training session.
        self.trainUpdate = [self.wMatrix.assign_add(self.wAdjust),
                            self.vBias.assign_add(self.vBAdjust),
                            self.hBias.assign_add(self.hBAdjust)]

    #Load training data.
    def LoadTrainingSet(self, directory):

        #Initialize blank training set.
        self.trainDataset = []

        if not(directory is not None and directory and os.path.isdir(directory)):
            print("Can't load training data - no valid directory specified!")
            return False

        #Use Glob to grab all the MIDI files in the chosen directory.
        fileset = glob.glob("{}/*.mid*".format(directory))

        self.midi.maxLength = DEFAULT_TIMESTEPS * 3;

        #Parse every file and convert it to a feature vector.
        for midifile in tqdm(fileset):
            try:
                fv = num.array(self.midi.MIDItoFV(midifile))

                if num.array(fv).shape[0] > DEFAULT_TIMESTEPS * 2:
                    self.trainDataset.append(fv)
            except Exception as e:
                print(e)

        print("Loaded {} MIDI files for training.".format(len(self.trainDataset)))

        return True

    #Train the network!
    def Train(self, event, saveDir):

        if not len(self.trainDataset) > 0:
            print("Can't train without any data!")
            return False

        #Sort out our persistence strategy.
        #Default to save in the temporary directory (caches last trained model).
        modelSaveLoc = MODEL_SAVE_LOC
        saveTmp = False

        #If a valid alternate directory is provided, use that.
        if saveDir is not None and saveDir and os.path.isdir(saveDir) and os.listdir(saveDir) == []:
            modelSaveLoc = saveDir
            os.rmdir(saveDir)
            saveTmp = True
            print("Model will be saved to " + modelSaveLoc)

        #Remove the existing cached model.
        #Tensorflow won't let us save to an existing directory.
        shutil.rmtree(MODEL_SAVE_LOC)
        netBuilder = tf.saved_model.builder.SavedModelBuilder(modelSaveLoc)

        #If an alternate is provided, we still want to write to the tmp cache.
        if saveTmp:
            tmpNetBuilder = tf.saved_model.builder.SavedModelBuilder(MODEL_SAVE_LOC)

        with tf.Session() as session:

            session.run(tf.global_variables_initializer())

            #Epoch count is configured earlier.
            #TQDM will let us monitor progression in the console.
            for epoch in tqdm(range(self.epochs)):

                #For every training FV (song) we have...
                for tData in self.trainDataset:
                    #Convert/reshape the FV to work with Tensorflow.
                    tData = num.array(tData)
                    tData = tData[:int(num.floor(tData.shape[0] // self.timesteps) * self.timesteps)]
                    tData = num.reshape(tData, [tData.shape[0] // self.timesteps, tData.shape[1] * self.timesteps])

                    #Run through the current example, chunking the data according to our batch size.
                    for i in range(1, len(tData), self.batchSize):
                        tr_x = tData[i:i + self.batchSize]
                        session.run(self.trainUpdate, feed_dict={self.notedata: tr_x})

            #Prep our builders to save the model.
            netBuilder.add_meta_graph_and_variables(session, ["RBMNet"])

            if saveTmp:
                tmpNetBuilder.add_meta_graph_and_variables(session, ["RBMNet"])

        #Save and/or cache our model.
        netBuilder.save()

        if saveTmp:
            tmpNetBuilder.save()

        print("Model stored in " + modelSaveLoc)
        return True

    #Check to see if a cached model is available.
    def IsTmpModelStored(self):
        modelLoadCheck = MODEL_SAVE_LOC + "/saved_model.pb"
        return os.path.isfile(modelLoadCheck)

    #Load a model from saved state/cache and generate new music.
    def Generate(self, event, loadDir, saveDir):

        #Initialize our loading directory.
        sampleSaveLoc = SAMPLE_LOC

        if saveDir is not None and saveDir and os.path.isdir(saveDir):
            sampleSaveLoc = saveDir

        modelLoadLoc = MODEL_SAVE_LOC

        if loadDir is not None and loadDir and os.path.isdir(loadDir):
            modelLoadLoc = loadDir

        modelLoadCheck = modelLoadLoc + "/saved_model.pb"

        #Check to make sure the model we're attempting to load is there.
        if not os.path.isfile(modelLoadCheck):
            print("Can't generate with no network available!")
            return False

        print("Loading model from " + modelLoadLoc + "...")
        tf.reset_default_graph()

        with tf.Session() as session:

            session.run(tf.global_variables_initializer())

            #Restore our graph state.
            tf.saved_model.loader.load(session, ["RBMNet"], modelLoadLoc)

            #To my knowledge, this really is the best way to reinitialize our Tensorflow variable references. Sad!
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

            #Sample our network.
            #This will generate a multidimensional array containing reconstructed visible layer data (i.e., songs)
            #for every sample we want to generate (the number of samples is customizable in the UI).
            sample = Gibbs(k=1, x=self.notedata, wMatrix=self.wMatrix, hBias=self.hBias, vBias=self.vBias).eval(
                session=session, feed_dict={self.notedata: num.zeros((self.genSample, self.vNodes))})

            #Reshape and convert our data back into MIDI format for each sample generated.
            for i in range(sample.shape[0]):
                if not any(sample[i, :]):
                    continue
                S = num.reshape(sample[i, :], (self.timesteps, 2 * self.notespan))
                self.midi.FVtoMIDI(S, sampleSaveLoc + "/OutputSample-{}".format(i))

        print("Saved samples to " + sampleSaveLoc)
        return True