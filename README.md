# spectral-centroid-splitter

Splits a uncompressed wav file into small fractions and sorts them after a [spectral-centroid](https://en.wikipedia.org/wiki/Spectral_centroid) analysis.

### Requirements

* python 2.7
* matplotlib 1.4.3
* numpy 1.9.2

### Usage

'''
usage: spectral-centroid-splitter.py [-h] [--duration [ms]] [--sort-grid [hz]]
                                     [--output-dir path]
                                     input-wav-file

positional arguments:
  input-wav-file     input wave file path (uncompressed)

optional arguments:
  -h, --help         show this help message and exit
  --duration [ms]    duration of one wave fraction (in milliseconds)
  --sort-grid [hz]   sort grid after spectral analysis (in hz steps)
  --output-dir path  output directory path
'''

### Example

`python spectral-centroid-splitter.py test.wav --duration 150 --sort-grid 50 --output-dir .`

'''
Name                  "test"
Duration              4.595021s
Sample-Rate           16bit
Frame-Rate            48000hz
Fraction-Size-Frames  2397
Fraction-Sort-Grid    100
Fraction-Duration     0.050000s
Fractions-Count       92
'''

Example-Output @ `out/1300/test_out_2.png` (spectral centroid at around 1300hz)

![Example](https://github.com/marmorkuchen-net/spectral-centroid-splitter/blob/master/example.png)
