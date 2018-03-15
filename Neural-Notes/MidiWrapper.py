import midi
import numpy as num

#Thanks to the excellent online code reference for using Python-MIDI
#to parse MIDI event lists for feature extraction:
#http://danshiebler.com/2016-08-10-musical-tensorflow-part-one-the-rbm/

class NNMidiUtility:

    def __init__(self, lowBound=36, highBound=85):

        #Establish the range of notes we'll consider in the model.
        self.lowBound = lowBound
        self.highBound = highBound
        self.notespan = highBound - lowBound

    def MIDItoFV(self, filename):

        #print(filename)
        #Grab the MIDI file as an event list from Python-MIDI.
        midiEvents = midi.read_midifile(filename)

        #Initialize lists for keeping track of...tracks.
        trackTimes = [track[0].tick for track in midiEvents]
        trackPos = [0 for track in midiEvents]

        #Initialize our feature vector.
        #To start, this will be built as a matrix.
        fv = []

        curState = [[0,0] for n in range(self.notespan)]
        prevState = curState

        #Add a sample of silence to the beginning of the feature vector.
        fv.append(curState)

        keepParsing = True
        tick = 0

        while keepParsing:

            #Do we need to register a new note state?
            if tick % (midiEvents.resolution / 4) == (midiEvents.resolution / 8):
                prevState = curState
                curState = [[prevState[n][0],0] for n in range(self.notespan)]
                fv.append(curState)

            #Iterate over tracks and update note data.
            for t in range(len(trackTimes)):

                if not keepParsing:
                    break

                while trackTimes[t] == 0:

                    track = midiEvents[t]
                    tPos = trackPos[t]
                    tEvent = track[tPos]

                    if isinstance(tEvent, midi.NoteEvent):
                        #Ignore out-of-range notes.
                        if (tEvent.pitch < self.lowBound) or (tEvent.pitch >= self.highBound):
                            pass
                        #Register note "presses".
                        elif isinstance(tEvent, midi.NoteOnEvent) and tEvent.velocity != 0:
                            curState[tEvent.pitch - self.lowBound] = [1,1]
                        #Register note "releases".
                        else:
                            curState[tEvent.pitch - self.lowBound] = [0,0]
                    #Ignore non-4 time signatures FOR NOW.
                    elif isinstance(tEvent, midi.TimeSignatureEvent):
                        if tEvent.numerator not in (2, 4):
                            print("SKIPPING FILE!")
                            keepParsing = False
                            break

                    if tPos + 1 < len(track):
                        trackTimes[t] = track[tPos + 1].tick
                        trackPos[t] += 1
                    else:
                        trackTimes[t] = None

                if trackTimes[t] is not None:
                    trackTimes[t] -= 1

            if all(tTime is None for tTime in trackTimes):
                break

            tick += 1

        fvArray = num.array(fv)
        fv = num.hstack((fvArray[:,:,0], fvArray[:,:,1]))
        fv = num.asarray(fv).tolist()

        return fv

    def FVtoMIDI(self, fv, filename):

        #Process our feature vector/matrix.
        fv = num.array(fv)

        if len(fv.shape) != 3:
            fv = num.dstack((fv[:,:self.notespan], fv[:,self.notespan:]))

        fv = num.asarray(fv)

        midiEvents = midi.Pattern()
        track = midi.Track()
        midiEvents.append(track)

        testTickScale = 55

        lTime = 0
        prevState = [[0,0] for n in range(self.notespan)]

        for fTime, fState in enumerate(fv + [prevState[:]]):

            notesOn = []
            notesOff = []

            for n in range(self.notespan):

                note = fState[n]
                prevNote = prevState[n]

                if prevNote[0] == 1:
                    if note[0] == 0:
                        notesOff.append(n)
                    elif note[1] == 1:
                        notesOn.append(n)
                elif note[0] == 1:
                    notesOn.append(n)

            for note in notesOff:

                track.append(midi.NoteOffEvent(tick = (fTime - lTime) * testTickScale,
                                               pitch = note + self.lowBound))
                lTime = fTime

            for note in notesOn:

                track.append(midi.NoteOnEvent(tick = (fTime - lTime) * testTickScale,
                                              velocity = 40,
                                              pitch = note + self.lowBound))
                lTime = fTime

            prevState = fState

        track.append(midi.EndOfTrackEvent(tick = 1))
        midi.write_midifile("{}.midi".format(filename), midiEvents)

        print("Wrote MIDI file: " + "{}.midi".format(filename))
