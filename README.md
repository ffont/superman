# Superman

*Superman* is an *audio mosaicing* command line utility to make an alternative version of a *target* audio file by concatenating chunks from a number of other *source* audio files. This code is adapted from a lab assignment of the [Master in Sound and Music Computing](https://www.upf.edu/web/smc) (of the [Music Technology Group](https://www.upf.edu/web/mtg)) which I did in 2009-2010. Original code provided in the lab assignment was by [Marteen de Boer](https://www.linkedin.com/in/maarten-de-boer-a508591/).


## Installation

This is a Python script that has the following Python requirements:

 * `essentia`
 * `scikits.audiolab`
 * `scikits.samplerate`
 * `numpy`

Essentia should be installed separately following [these instructions](http://essentia.upf.edu/documentation/installing.html). The other requirements should be installed by running:

	pip install numpy scikits.audiolab scikits.samplerate


## Basic instructions

 1. Put target audio file in same folder as `superman.py`
 2. Put source audio files inside a folder called `sources` next to `superman.py`
 3. Run and wait until the output file is created:

	```
	python superman.py TARGET_FILENAME --tempo TEMPO

	e.g.
	python superman.py 'target part 1.aif' --tempo 90.0
	```

Both the target file and the source files will be sliced and analyzed at chunks corresponding to a beat of the indicated tempo. Analysis of target and source files is stored in a file so that the next time the algorithm is run files don't need to be analyzed again. However, if tempo is changed, the new analysis for the new tempo will need to be computed.

The longer the target/source files are (and the more source files), the longer it will take to build the output... For long targets and long sources (i.e. minutes long), generating the output can take hours...


## Extra options

To do quick tests use the `--length` option to indicate how many seconds of the target you want reconstructed. Use it like:

	python superman.py TARGET_FILENAME --tempo TEMPO --length LENGTH

	e.g.
	python superman.py 'target part 1.aif' --tempo 90.0 --length 10

To write the output "score" in a *.csv* file that you can open with Excel, etc., add the option `--write_score`. Use it like:

	python superman.py TARGET_FILENAME --tempo TEMPO --write_score

You can specify a name for the output file using the option `--out_filepath`. If not specified, a default name will be chosen. Output file will be PCM WAV file, if extension *.wav* is not indicated as part of the out filepath, it will be automatically added. Use it like:

	python superman.py TARGET_FILENAME --tempo TEMPO --out_filepath OUT_FILEPATH

	e.g.
	python superman.py 'target part 1.aif' --tempo 90.0 --out_filepath 'my_output_tempo_90.wav'

You can add some randomization into the output (i.e. not always using the closest source unit for a given target) by using the option `--random`. In this way consecutive runs of the same build (i.e. same target and same sources) won't produce the same results. Use it like:

	python superman.py TARGET_FILENAME --tempo TEMPO --random RANDOM

	e.g.
	python superman.py 'target part 1.aif' --tempo 90.0 --random 0.3

For even more options, run the help command and see what's in there:

	python superman.py --help


NOTE: the time remaining indicator is only to be trusted after some iterations have passed (~10 approx).
