import sys
import os.path
import json
import argparse

import wave

import numpy
import matplotlib.pyplot as pyplot


# utilities

def separator(len=45):

    sep_str = ''

    for i in range(0, len):
        sep_str += '='

    return sep_str


# wave file handling

class Wave(object):

    def __init__(self, x, n, frame_rate, sample_rate, channel_num):

        self.data = x
        self.len = n

        self.duration = float(n) / frame_rate
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channel_number = channel_num

    def as_numpy_arr(self):

        arr = ''.join(self.data)
        arr = numpy.fromstring(
            arr, numpy.dtype('int%i' % self.sample_rate))
        arr = numpy.reshape(arr, (-1, self.channel_number))

        return arr


def open_wav(file_path):

    wav = wave.open(file_path, 'r')

    data = []
    n = wav.getnframes()

    for i in range(0, n):
        data.append(wav.readframes(1))

    wav.close()

    return Wave(
        x=data,
        n=n,
        frame_rate=wav.getframerate(),
        sample_rate=wav.getsampwidth() * 8,
        channel_num=wav.getnchannels()
    )


def write_wav(wav, file_path):

    output_wav = wave.open(file_path, 'w')

    output_wav_params = (
        wav.channel_number,
        wav.sample_rate / 8,
        wav.frame_rate,
        0, 'NONE', 'not compressed'
    )

    output_wav.setparams(output_wav_params)
    output_wav.writeframes(wav.as_numpy_arr())
    output_wav.close()


# json file handling

class NumpyAwareJSONEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


def write_json(file_name, json_data):

    json_file = open('%s.json' % file_name, 'w')
    json_file.write(json_data)
    json_file.close()


# signal analysis

def fft(wav):

    data = wav.as_numpy_arr()

    snd = data / (2.**(wav.sample_rate - 1))
    signal = snd[:, 0]

    # fft

    n = len(signal)
    p = numpy.fft.rfft(signal)

    unique_points = numpy.ceil((n + 1) / 2.0)

    p = p[0:unique_points]
    p = abs(p)

    p = p / float(n)
    p = p**2

    if n % 2 > 0:
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) - 1] = p[1:len(p) - 1] * 2

    frequency_array = numpy.arange(
        0, unique_points, 1.0) * (float(wav.frame_rate) / n)

    return {
        'n': n,
        'x': frequency_array,
        'y': p
    }


def spectral_centroid(fft_data, fft_len, frame_rate):

    magnitudes = numpy.abs(fft_data)
    freqs = numpy.abs(
        numpy.fft.fftfreq(fft_len, 1.0 / frame_rate)[:fft_len // 2 + 1])

    return numpy.sum(magnitudes * freqs) / numpy.sum(magnitudes)


# fraction class

class AudioFraction():

    def __init__(self, fraction_index, wav):

        self.fraction_index = fraction_index
        self.wav = wav

        # analyse this fraction

        self.fft = fft(self.wav)
        self.spectral_centroid = spectral_centroid(
            self.fft['y'], self.fft['n'], self.wav.frame_rate)

    def save_as_json(self, file_name):

        json_data = json.dumps({
            'x': self.fft['x'],
            'y': 10 * numpy.log10(self.fft['y'])
        }, cls=NumpyAwareJSONEncoder)

        write_json(file_name, json_data)

    def save_as_wav(self, file_name):

        write_wav(self.wav, '%s.wav' % file_name)

    def save_as_png(self, file_name):

        pyplot.plot(
            self.fft['x'] / 1000, 10 * numpy.log10(self.fft['y']), color='k')

        pyplot.xlabel('Frequency (kHz)')
        pyplot.ylabel('Power (dB)')

        pyplot.savefig('%s.png' % file_name, bbox_inches='tight')

        pyplot.clf()


class AudioFractionFactory():

    def __init__(self, base_name, output_dir, wav, fraction_duration, sort):

        self.wav = wav
        self.output_dir = output_dir
        self.sort_grid = sort

        fraction_count = int(
            numpy.ceil(wav.duration / fraction_duration))

        self.fraction = {
            'duration': fraction_duration,
            'count': fraction_count,
            'wav_len': int(numpy.ceil(float(wav.len / fraction_count))),
            'file_name': base_name
        }

    def print_parameters(self):

        print 'PARAMETERS'
        print separator()

        print 'Name\t\t\t"%s"\nDuration\t\t%fs' % (
            self.fraction['file_name'],
            self.wav.duration
        )

        print 'Sample-Rate\t\t%ibit\nFrame-Rate\t\t%ihz' % (
            self.wav.sample_rate,
            self.wav.frame_rate
        )

        print 'Fraction-Size-Frames\t%i\nFraction-Sort-Grid\t%i' % (
            self.fraction['wav_len'],
            self.sort_grid
        )

        print 'Fraction-Duration\t%fs\nFractions-Count\t\t%i' % (
            self.fraction['duration'],
            self.fraction['count']
        )

        print separator()

    def generate(self):

        for i in range(0, self.fraction['count']):

            start = i * self.fraction['wav_len']
            end = start + self.fraction['wav_len']

            fraction_wav = Wave(
                x=self.wav.data[start:end],
                n=self.fraction['wav_len'],
                frame_rate=self.wav.frame_rate,
                sample_rate=self.wav.sample_rate,
                channel_num=self.wav.channel_number
            )

            fraction = AudioFraction(
                fraction_index=i,
                wav=fraction_wav
            )

            # sort file in directories

            sort_dir = numpy.ceil(fraction.spectral_centroid)
            sort_dir = int(round(sort_dir / self.sort_grid) * self.sort_grid)
            base_dir = '%s/%s' % (self.output_dir, sort_dir)

            file_name = '%s/%s_out_%i' % (
                base_dir,
                self.fraction['file_name'],
                i + 1
            )

            if not os.path.isdir(base_dir):
                os.makedirs(base_dir)

            # write files

            print 'WRITE fraction "%s" [%i] (%i of %i)' % (
                self.fraction['file_name'],
                sort_dir,
                i + 1,
                self.fraction['count']
            )

            fraction.save_as_json(file_name)
            fraction.save_as_png(file_name)
            fraction.save_as_wav(file_name)


# main

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Splits a wav file into small fractions \
        and sorts them after a spectral-centroid analysis'
    )

    parser.add_argument('input', metavar='input-wav-file',
                        type=str,
                        help='input wave file path (uncompressed)'
                        )

    parser.add_argument('--duration', metavar='ms', type=int, nargs='?',
                        default=50,
                        help='duration of one wave fraction (in milliseconds)')

    parser.add_argument('--sort-grid', metavar='hz', type=int, nargs='?',
                        default=100,
                        help='sort grid after spectral analysis (in hz steps)')

    parser.add_argument('--output-dir', metavar='path', type=str,
                        default='./out', help='output directory path')

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print 'ERROR invalid input wav file name'
        sys.exit()

    if not os.path.isdir(args.output_dir):
        print 'creating new output directory "%s"\n' % args.output_dir
        os.makedirs(args.output_dir)

    wav = open_wav(args.input)

    factory = AudioFractionFactory(
        base_name=args.input.rsplit('.', 1)[0],
        output_dir=args.output_dir,
        wav=wav,
        sort=args.sort_grid,
        fraction_duration=(args.duration / 1000.0)
    )

    factory.print_parameters()
    factory.generate()
