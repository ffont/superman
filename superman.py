from essentia.standard import *
from scikits.samplerate import resample
from math import sqrt, ceil
from random import random
import numpy
import scikits.audiolab as audiolab
import sys
import essentia
import os
import json
import argparse
import time
import datetime

parser = argparse.ArgumentParser(description='Make an alternative version of a "target" audio file by concatenating chunks from a number of other "source" audio files.')
parser.add_argument('target_file', type=str, help='Filepath of the target file.')
parser.add_argument('--tempo', type=float, default=90.0, help='Splits target (and source) files in units with length equal to a beat of this tempo (default is 90.0).')
parser.add_argument('--random', type=float, default=0.0, help='Amount of randomization in unit selection (default is 0.0).')
parser.add_argument('--overlap_percentage', dest='overlap_percentage', type=float, default=0.5, help='Overlapping percentage of target units. Should be in range [0.0, 1.0], default is 0.5.')
parser.add_argument('--overlap_percentage_source', dest='overlap_percentage_source', type=float, default=0.5, help='Overlapping percentage of source units. Should be in range [0.0, 1.0], default is 0.5.')
parser.add_argument('--sources_dir', dest='sources_dir', type=str, default='sources/', help='Directory where to look for source files.')
parser.add_argument('--out_filepath', dest='out_filepath', type=str, default=None, help='Output path of the generated audio file.')
parser.add_argument('--length', dest='length', type=float, default=None, help='Length of the output in seconds.')
parser.add_argument('--force_analyze', dest='force_analyze', action="store_true", help='Force re-analyze all sources/targets.')
parser.add_argument('--write_score', dest='write_score', action="store_true", help='Write "score" of chosen units.')
parser.add_argument('--sample_rate', type=int, default=44100, help='Sample rate of the files (all should be the same, default is 44100).')



def time_stats(done, total, starttime):
    nowtime = time.time()
    position = done*1.0 / total
    duration = round(nowtime - starttime)
    durdelta = datetime.timedelta(seconds=duration)
    remaining = round((duration / position) - duration)
    remdelta = datetime.timedelta(seconds=remaining)

    return str(durdelta), str(remdelta)


def analyze(filepath, frame_size=1024, hop_size=512, sample_rate=44100):
    loader = MonoLoader(filename=filepath, sampleRate=sample_rate)
    audio = loader()
    w = Windowing(type = 'blackmanharris62')
    peaks = SpectralPeaks(maxPeaks=10000, maxFrequency=5000, minFrequency=40)
    hpcp = HPCP(harmonics=8, maxFrequency=5000, minFrequency=40, size=36, windowSize=0.5)
    energy = Energy()
    hpcps = []
    energys = []
    indexes = []
    idx = 0
    for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True, lastFrameToEndOfFile=False):
        e = energy(w(frame))
        energys.append(e)
        h = hpcp(*peaks(w(frame)))
        h = [float(e) for e in h]
        hpcps.append(h)  # Convert to python list so it can be serialized to JSON
        indexes.append((idx, idx + frame_size))
        idx += hop_size
    return {
		'energy': energys,
		'hpcp': hpcps,
		'n_frames': len(energys),
		'indexes': indexes,
		'filepath': filepath,
	}


def analyze_file_or_load_analysis(filepath, frame_size, hop_size, sample_rate):
	analysis_filepath = filepath + '.analysis_%i_%i_%i.json' % (frame_size, hop_size, sample_rate)
	if not os.path.exists(analysis_filepath) or force_analyze:
		# If analysis does not exist for given (frame_size, hop_size, sample_rate), run it
		print 'Analyzing:', filepath
		file_analysis = analyze(filepath, frame_size=frame_size, hop_size=hop_size, sample_rate=sample_rate)
		json.dump(file_analysis, open(analysis_filepath, 'w'))
	else:
		file_analysis = json.load(open(analysis_filepath))
	return file_analysis


def calcHPCPDist(tgtHPCP,srcHPCP):
    # calc the distance between target and source hpcp descriptors, at different circular shifts
    nBins = len(tgtHPCP)
    if nBins != len(srcHPCP):
        raise "cannot compare arrays of different length"

    if nBins != 36:
        raise "expected 36 HPCP bins per frame"

    maxShiftBin = 18 
    bestShiftBin = 0.0
    bestDist = 100000.0

    startShiftBin = -maxShiftBin
    endShiftBin = maxShiftBin

    for shiftBin in range(startShiftBin,endShiftBin+1):
        # calculate the distance between ref and cmp,
        # with a shift of shiftBin bins
        dist = 0.0
        corr = 0.0
        for bin in range(0,nBins):
            # for each bins, at the current shift, add the distance
            # to distance
            tgtBin = bin
            srcBin = (bin + shiftBin + nBins) % nBins

            tgtVal = tgtHPCP[tgtBin]
            srcVal = srcHPCP[srcBin]

            d = tgtVal - srcVal
            dist += d*d
         #corr += (tgtVal-.5) * (srcVal-.5)

        dist = sqrt(dist)
        #dist = (nBins*.5*.5)-corr

        # was this the best so far?
        if dist < bestDist or (dist == bestDist and abs(shiftBin) < abs(bestShiftBin)):
            # yes: we remember it
            bestDist = dist
            bestShiftBin = shiftBin

    # return the best shift and distance found
    return (bestShiftBin,bestDist)

def morph(source_analysis, target_analysis, outFilepath, windowsize=8192, hopsize=2048, max_output_length=None, rand_amount=0.3):

    starttime = time.time()

    output_score = []

    # load target audio file and analysis
    targetsf = audiolab.Sndfile(target_analysis['filepath'],'r')
    samplerate = targetsf.samplerate
    t_energy =  numpy.array(target_analysis['energy'])
    t_hpcps = numpy.array([numpy.array(e) for e in target_analysis['hpcp']])

    # load source audio files and analysis
    src_audio_files = dict()
    s_energy = numpy.array([])
    s_hpcps = numpy.zeros((0,t_hpcps.shape[1]))
    s_analysis_index = []
    for element in source_analysis:
        # load audio file
        sourcesf = audiolab.Sndfile(element['filepath'],'r')
        file_frames = sourcesf.read_frames(sourcesf.nframes)
        src_audio_files[element['filepath']] = file_frames

        # concatenate analysis in single arrays
        s_energy = numpy.append(s_energy, numpy.array(element['energy']))
        hpcps = numpy.array([numpy.array(e) for e in element['hpcp']])
        s_hpcps = numpy.append(s_hpcps, hpcps, axis=0)
        for frame_bounds in element['indexes']:
            s_analysis_index.append((element['filepath'], frame_bounds[0], frame_bounds[1]))

    #init variables, generate the window(s) for the overlap-add
    nchannels = targetsf.channels
    window = 0.5-numpy.cos(2.*numpy.pi*numpy.arange(0,windowsize)/float(windowsize))/2
    stereowindow = numpy.zeros((windowsize,nchannels))
    for i in range(0,nchannels):
        stereowindow[:,i] = window
    
    #print 'Source analysis array shapes:', s_energy.shape, s_hpcps.shape
    #print 'Target analysis array shapes:', t_energy.shape, t_hpcps.shape

    #######################################################################

    # load energy and hpcps: target
    tgtEs = [[x] for x in t_energy]
    tgtHPCPs = t_hpcps
    ttimes = []
    t = 0
    for x in tgtEs:
        ttimes.append(t)
        t += hopsize 
               
    # load energy and hpcps: source
    srcEs = [[x] for x in s_energy]
    srcHPCPs = s_hpcps
    stimes = []
    t = 0
    for x in srcEs:
        stimes.append(t)
        t += hopsize 

    # create empty output arrays
    outsize = targetsf.nframes + windowsize
    out = numpy.zeros((outsize,nchannels))
    lastBestIndex = -1

    # go through all target units and find best match
    if max_output_length:
        max_analysis_frames = int(ceil((max_output_length * samplerate) / hopsize))
    else:
        max_analysis_frames = len(tgtHPCPs)
      
    for tgtHPCPIndex in range(0,max_analysis_frames):
        _, remaining = time_stats(tgtHPCPIndex + 1, max_analysis_frames, starttime)

        sys.stdout.write('\rFinding units for frames [%i/%i] - %s remaining' % (tgtHPCPIndex + 1, max_analysis_frames, remaining))
        sys.stdout.flush()

        tgtHPCP = tgtHPCPs[tgtHPCPIndex]
        # and for each, look for the source HPCP that matches best
        bestIndex = 0
        bestShiftBin = 0
        bestDist = 100000.0
        bestEDist = 100000.0; # FOR ENERGY
        #bestEindex = 0; # FOR ENERGY
        
        for srcHPCPIndex in range(0,len(srcHPCPs)):

            # COMPUTE HPCP DIST
            srcHPCP = srcHPCPs[srcHPCPIndex]
            (shiftBin,hpcpDist) = calcHPCPDist(tgtHPCP,srcHPCP)
            hpcpDist += random()*rand_amount # add some random to introduce more variation
            dist = hpcpDist

            # COMPUTE ENERGY DIST
            tgtE=tgtEs[tgtHPCPIndex]
            srcE=srcEs[srcHPCPIndex]
            eDist=abs(tgtE[0]-srcE[0])
            #eIndex=srcHPCPIndex
            #if eDist < bestEDist :
            #    bestEindex = srcHPCPIndex
            #    bestEDist = eDist

            # was this the best so far?
            # (meaning: shorter distance, or if the same distance, less shifting
            # in case of equal HPCP dist, we select less shift difference, and in case of equal shift difference we select les energy difference
            if dist < bestDist or (dist == bestDist and abs(shiftBin) < abs(bestShiftBin)) or (dist == bestDist and abs(shiftBin) == abs(bestShiftBin) and eDist < bestEDist):
                #if eDist < bestEDist:
                ###################################
                # Avoid repeating frames
                if srcHPCPIndex == lastBestIndex :
                    # If the current frame is the same as the last one, we bypass the selection and go to the next.
                    pass #print "repetition avoided"
                else:
                    bestIndex = srcHPCPIndex
                    bestDist = dist
                    bestShiftBin = 0#shiftBin # NEVER SHIFT!
                    bestEDist = eDist

            ###################################
            # Combine ENERGY and HPCP information: we take the closer index to the target (just as an example)
            #hpcpDiff=abs(tgtHPCPIndex-bestIndex)
            #eDiff=abs(tgtHPCPIndex-bestEindex)
            #usedEnergy=0;
            #if hpcpDiff > eDiff :
            #    bestIndex = bestEindex
            #    usedEnergy = 1
            #else:
            #    bestIndex = bestIndex
            #    usedEnergy = 0
            lastBestIndex = bestIndex

        # add info to the score
        s_filename, srcPos_0, srcPos_1 = s_analysis_index[bestIndex]
        time_target_unit = (tgtHPCPIndex * hopsize * 1.0)/samplerate
        time_source_unit = (srcPos_0 * 1.0)/samplerate
        output_score.append((time_target_unit, s_filename, time_source_unit))

        # get corresponding audio unit from source
        src_data_array = src_audio_files[s_filename]
        #print s_filename, srcPos_0, srcPos_1, src_data_array.shape
        sbuf = numpy.zeros((windowsize,nchannels))
        for channel in range(nchannels):
            if len(src_data_array.shape) == 2:
                data = src_data_array[srcPos_0:srcPos_1, min(channel, src_data_array.shape[1])]
                sbuf[:,channel] = numpy.pad(data, (0, len(sbuf) - len(data)), 'constant', constant_values=0.0)
            else:
                data = src_data_array[srcPos_0:srcPos_1]
                sbuf[:,channel] = numpy.pad(data, (0, len(sbuf) - len(data)), 'constant', constant_values=0.0)
        sbuf *= stereowindow

        # change the pitch of the selected audio according to the shift 
        resampling = 2.**(float(bestShiftBin)/float(len(srcHPCPs[bestIndex])))
        resamplingtype = 'linear'
        rsbufs = []
        for channel in range(nchannels):
            rsbufs.append(resample(sbuf[:,channel], resampling, resamplingtype))
        rsbuf = numpy.zeros((len(rsbufs[0]),nchannels))
        for channel in range(nchannels):
            rsbuf[:,channel] = rsbufs[channel]
        rl = len(rsbuf)
        tgtPos = int(ttimes[tgtHPCPIndex])
        #tgtPos -= rl/2;
        #tgtPos = max(0,min(outsize-rl,tgtPos))
        
        # add the audio to the output buffer
        out[tgtPos:tgtPos+rl] += rsbuf*0.5

    # compute energy of target and output audios
    # and modify the output audio by applying an envelope so it follows the target audio
    print '\nApplying energy envelope...'
    winsize = 2048+1;
    hwinsize = (winsize-1)/2;
    hopsize = 128;
    tmp = numpy.hanning(winsize)
    synthwin = numpy.zeros((winsize,nchannels))
    for channel in range(nchannels):
        synthwin[:,channel] = tmp
    out2 = numpy.zeros((len(out),nchannels))
    out2env = numpy.zeros((len(out),nchannels))
    if max_output_length is not None:
        outsize = int(max_output_length*samplerate)

    for pos in range(0,outsize,hopsize):
        targetsf.seek(max(0,min(targetsf.nframes-winsize,pos-hwinsize)))
        tmp = targetsf.read_frames(winsize)
        tbuf = numpy.zeros((winsize,nchannels))
        for channel in range(nchannels):
            if len(tmp.shape) == 2:
                tbuf[:,channel] = tmp[:, min(channel, tmp.shape[1])]
            else:
                tbuf[:,channel] = tmp

        tmp = max(0,min(outsize-winsize,pos-hwinsize))
        obuf = out[tmp:tmp+winsize].copy()
        obuf *= synthwin
        tbuf *= synthwin

        tenergy = numpy.sum( numpy.power(tbuf,2) )
        oenergy = numpy.sum( numpy.power(obuf,2) )
        obuf *= sqrt(tenergy/(oenergy+.000000001))
        out2[tmp:tmp+winsize] += obuf
        out2env[tmp:tmp+winsize] += synthwin

    out2env += 0.0000000001  # this is to avoid divide warning
    out2 /= out2env
    out2 = numpy.clip(out2,-1,1)
    if max_output_length is not None:
        out2 = out2[:int(max_output_length*samplerate)]

    format = audiolab.Format('wav','pcm16')
    sfout = audiolab.Sndfile(outFilepath,'w',format,nchannels,targetsf.samplerate)
    sfout.write_frames(out2)
    sfout.close()
    print 'Done! File written in:', outFilepath
    return output_score


# PARSE ARGS AND DO THE MORPHING
args = parser.parse_args()
tempo = args.tempo
target_filepath = args.target_file
sample_rate = args.sample_rate
force_analyze = args.force_analyze
frame_size = int(round(sample_rate * 60.0 / tempo))

if args.overlap_percentage < 0.0:
	raise Exception('--overlap_percentage should be in range [0.0, 1.0]')
elif args.overlap_percentage > 1.0:
	raise Exception('--overlap_percentage should be in range [0.0, 1.0]')
hop_size = int(round(frame_size * (1.0 - args.overlap_percentage)))
if hop_size < 1:
	hop_size = 1

if args.overlap_percentage_source < 0.0:
	raise Exception('--overlap_percentage_source should be in range [0.0, 1.0]')
elif args.overlap_percentage > 1.0:
	raise Exception('--overlap_percentage_source should be in range [0.0, 1.0]')
hop_size_source = int(round(frame_size * (1.0 - args.overlap_percentage_source)))
if hop_size_source < 1:
	hop_size_source = 1


max_output_length = None
if args.length is not None:
	max_output_length = abs(args.length)
write_score = args.write_score
SOURCE_DIR = args.sources_dir

# Analyze target
target_analysis = analyze_file_or_load_analysis(target_filepath, frame_size, hop_size, sample_rate)
print 'Loaded analysis for target file (%i target units)' % target_analysis['n_frames']

# Analyze source files
source_analysis = []
for filename in os.listdir(SOURCE_DIR):
	filepath = os.path.join(SOURCE_DIR, filename)
	if filepath.endswith('.json') or filepath.endswith('.DS_Store'):
		continue
	file_analysis = analyze_file_or_load_analysis(filepath, frame_size, hop_size_source, sample_rate)
	source_analysis.append(file_analysis)
print 'Loaded analysis for %i source audio files (%i source units)' % (len(source_analysis), sum([value['n_frames'] for value in source_analysis]))

# Do the morphing
print 'Will do morphing for file "%s" at tempo %.2f' % (target_filepath, tempo)
date_label = datetime.datetime.today().strftime("%Y-%m-%d_%H.%M")
if args.out_filepath is None:
	out_filepath = target_filepath + '.morphed_%.2f.%s.wav' % (tempo, date_label)
else:
	out_filepath = args.out_filepath
	if not out_filepath.endswith('.wav'):
		out_filepath += '.wav'
score = morph(source_analysis, target_analysis, out_filepath, windowsize=frame_size, hopsize=hop_size, max_output_length=max_output_length, rand_amount=args.random)
if write_score:
	score_filename = out_filepath + '.score.csv'
	fid = open(score_filename, 'w')
	fid.write('Target time;Source time;Source file\n')
	for (target_time, source_filename, source_time) in score:
		fid.write('%.2fs;\t%.2fs;\t%s\n' % (target_time, source_time, source_filename))
	fid.close()
