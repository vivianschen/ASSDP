import pdb
import numpy as np
import scipy.signal
import scipy.misc
import librosa
import librosa.display
import soundfile as sf
import matplotlib
import matplotlib.pyplot as plt

from pychorus import find_and_output_chorus
from pychorus import create_chroma
from pychorus.similarity_matrix import TimeTimeSimilarityMatrix, TimeLagSimilarityMatrix, Line
import msaf
import sys
from dtw import dtw, accelerated_dtw
from scipy.spatial.distance import cdist
from math import isinf

# Denoising size in seconds
SMOOTHING_SIZE_SEC = 2.5

# Number of samples to consider in one chunk.
# Smaller values take more time, but are more accurate
N_FFT = 2**7

# For line detection
LINE_THRESHOLD = 0.15
MIN_LINES = 10
NUM_ITERATIONS = 20

# We allow an error proportional to the length of the clip
OVERLAP_PERCENT_MARGIN = 0.2

#chorus_start_sec = find_and_output_chorus("chorus_detect.wav", "test.wav", 7)

#chorus_find = find_chorus()


def local_maxima_rows(denoised_time_lag):
    """Find rows whose normalized sum is a local maxima"""
    row_sums = np.sum(denoised_time_lag, axis=1)
    divisor = np.arange(row_sums.shape[0], 0, -1)
    normalized_rows = row_sums / divisor
    local_minima_rows = scipy.signal.argrelextrema(normalized_rows, np.greater)
    return local_minima_rows[0]


def detect_lines(denoised_time_lag, rows, min_length_samples):
    """Detect lines in the time lag matrix. Reduce the threshold until we find enough lines"""
    cur_threshold = LINE_THRESHOLD
    for _ in range(NUM_ITERATIONS):
        line_segments = detect_lines_helper(denoised_time_lag, rows,
                                            cur_threshold, min_length_samples)
        if len(line_segments) >= MIN_LINES:
            return line_segments
        cur_threshold *= 0.95

    return line_segments


def detect_lines_helper(denoised_time_lag, rows, threshold,
                        min_length_samples):
    """Detect lines where at least min_length_samples are above threshold"""
    num_samples = denoised_time_lag.shape[0]
    line_segments = []
    cur_segment_start = None
    for row in rows:
        if row < min_length_samples:
            continue
        for col in range(row, num_samples):
            if denoised_time_lag[row, col] > threshold:
                if cur_segment_start is None:
                    cur_segment_start = col
            else:
                if (cur_segment_start is not None
                   ) and (col - cur_segment_start) > min_length_samples:
                    line_segments.append(Line(cur_segment_start, col, row))
                cur_segment_start = None
    return line_segments

def count_overlapping_lines(lines, margin, min_length_samples):
    """Look at all pairs of lines and see which ones overlap vertically and diagonally"""
    line_scores = {}
    for line in lines:
        line_scores[line] = 0

    # Iterate over all pairs of lines
    for line_1 in lines:
        for line_2 in lines:
            # If line_2 completely covers line_1 (with some margin), line_1 gets a point
            lines_overlap_vertically = (
                line_2.start < (line_1.start + margin)) and (
                    line_2.end > (line_1.end - margin)) and (
                        abs(line_2.lag - line_1.lag) > min_length_samples)

            lines_overlap_diagonally = (
                (line_2.start - line_2.lag) < (line_1.start - line_1.lag + margin)) and (
                    (line_2.end - line_2.lag) > (line_1.end - line_1.lag - margin)) and (
                        abs(line_2.lag - line_1.lag) > min_length_samples)

            if lines_overlap_vertically or lines_overlap_diagonally:
                line_scores[line_1] += 1

    return line_scores

def sorted_segments(line_scores):
    """Return the p line, sorted first by chorus matches, then by duration"""
    lines_to_sort = []
    for line in line_scores:
        lines_to_sort.append((line, line_scores[line], line.end - line.start))

    lines_to_sort.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return lines_to_sort

# def dtw(x,y):
#     distances = np.zeros((len(y), len(x)))
#
#     for i in range(len(y)):
#         for j in range(len(x)):
#             distances[i,j] = (x[j]-y[i])**2


file_name = "foo.wav"

chroma, song_wav_data, sr, song_length_sec = create_chroma("audio/" + file_name)

chorus_start_sec = find_and_output_chorus("audio/foo.wav", "chorus.wav", 15)

num_samples = chroma.shape[1]
time_time_similarity = TimeTimeSimilarityMatrix(chroma, sr)
time_lag_similarity = TimeLagSimilarityMatrix(chroma, sr)

#time_time_similarity.display()
print(msaf.get_all_label_algorithms())
print(msaf.get_all_boundary_algorithms())

#novelty based segmentation
#uses the foote or checkerboard kernel method of segmenting songs
sonified_file = "my_boundaries.wav"
#plot = True
boundaries, labels = msaf.process("audio/" + file_name, feature="mfcc", boundaries_id="foote",
                                    sonify_bounds=True,labels_id="fmc2d", out_sr=sr)

#audio = librosa.load(sonified_file, sr=sr)[0]

new_boundaries = []
new_labels = []
segment_nums = []
mfccs = []
idx = 0
for x in range(len(boundaries) - 1):
    if boundaries[x + 1] - boundaries[x] >= 3:
        print("segment found at {0:g} min {1:.2f} sec".format(
                boundaries[x] // 60, boundaries[x] % 60))
        segment_wav_data = song_wav_data[int(boundaries[x]*sr) : int(boundaries[x + 1]*sr)]

        mel_freq = librosa.feature.mfcc(segment_wav_data, sr)
        #print(mel_freq)
        #plt.imshow(mel_freq)
        #plt.savefig("segment_{}.png".format(idx))
        #librosa.display.specshow(mel_freq, x_axis='time')
        #plt.show()
        sf.write("segment_{}.wav".format(idx), segment_wav_data, sr)

        new_boundaries.append(boundaries[x])
        new_labels.append(labels[x])
        segment_nums.append(idx)
        mfccs.append(np.average(mel_freq, axis=0))
    idx += 1

#pdb.set_trace()

#min_length = min(length(mfccs[0]), mfccs[1])
def fastdtw(x, y, dist, warp=1):
    assert len(x)
    assert len(y)
    if np.ndim(x) == 1:
        x = x.reshape(-1, 1)
    if np.ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    return D1[-1, -1] / sum(D1.shape)


#intro - no vocal line
#verse - lower register, before the chorus
#pre-chorus - builds up to it, w most chord progression, dynamic changes
#chorus - stronger onseat beats
#bridge -
#outro -


# alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
# frames = open("labels/" + file_name + "_labels.txt", "w")
# for e in range(len(new_boundaries)):
#     if e < len(new_boundaries) - 1:
#         outer_bound = e+1
#         frames.write(str(new_boundaries[e]) + "\t" + str(new_boundaries[outer_bound]) + "\t" + alphabet[int(new_labels[e])] + "\n")
#     else:
#         frames.write(str(new_boundaries[e]) + "\t" + str(song_length_sec) + "\t" + alphabet[int(new_labels[e])] + "\n")
#

#sys.exit(1)

# time_time_similarity.display()
# time_lag_similarity.display()




chroma_sr = num_samples / song_length_sec
clip_length = 10
smoothing_size_samples = int(SMOOTHING_SIZE_SEC * chroma_sr)
time_lag_similarity.denoise(time_time_similarity.matrix,
                            smoothing_size_samples)

clip_length_samples = clip_length * chroma_sr

candidate_rows = local_maxima_rows(time_lag_similarity.matrix)


lines = detect_lines(time_lag_similarity.matrix, candidate_rows,
                     clip_length_samples)


if len(lines) == 0:
    print("No choruses were detected. Try a smaller search duration")
    sys.exit(-1)


line_scores = count_overlapping_lines(
    lines, OVERLAP_PERCENT_MARGIN * clip_length_samples,
    clip_length_samples)


choruses = sorted_segments(line_scores)

unsorted_chorus_times = []

for c in choruses:
    unsorted_chorus_times.append((c[0].start / chroma_sr, c[0].end / chroma_sr))

#pdb.set_trace()
unsorted_chorus_times.sort(key=lambda x: x[0])
#print(unsorted_chorus_times)

#merge overlapping intervals!!
chorus_times = []
chorus_times.append(unsorted_chorus_times[0])
for i in range(1, len(unsorted_chorus_times)):
    #pdb.set_trace()
    if (unsorted_chorus_times[i][0] - chorus_times[-1][0]) >= clip_length:
        chorus_times.append(unsorted_chorus_times[i])

chorus_wave_data = song_wav_data[int(chorus_times[0][0]*sr) : int(chorus_times[0][1]*sr)]
sf.write("chorus_segment.wav", chorus_wave_data, sr)
chorus_mfcc = np.average(librosa.feature.mfcc(chorus_wave_data, sr), axis=0)



structure_labels = [""] * len(new_labels)

["intro", "verse", "bridge", "outro"]
#"prechorus?"

#TODO: identify chorus section, then verse, then prechorus, then outro, intro, bridge, etc
euclidean_norm = lambda x, y: np.abs(x - y)
for x in range(len(new_boundaries)):
    chorus_sect = -1
    dist = fastdtw(mfccs[x], chorus_mfcc, dist=euclidean_norm)
    if dist < 2 or new_labels[x] == chorus_sect:
        chorus_sect = new_labels[x]
        structure_labels[x] = "chorus"
    print(dist)
    print()

# euclidean_norm = lambda x, y: np.abs(x - y)


for x in range(len(new_boundaries)):
    found_match = False
    for y in range(x + 1, len(new_boundaries)):
        if (new_labels[x] == new_labels[y]) and structure_labels[y] == "":
            found_match = True
            structure_labels[x] = "verse"
            structure_labels[y] = "verse"
    if found_match == False and structure_labels[x] == "":
        if x == 0:
            structure_labels[x] = "intro"
        elif x == (len(new_boundaries) - 1):
            structure_labels[x] = "outro"
        else:
            structure_labels[x] = "bridge"


# for x in range(len(new_boundaries)):
#     for y in range(x + 1, len(new_boundaries)):
#         dist = fastdtw(mfccs[x], mfccs[y], dist=euclidean_norm)
#         print(x + 1)
#         print(y + 1)
#         print(dist)
#         print()

print(new_boundaries)
print(new_labels)
print(structure_labels)



#euclidean_norm = lambda x, y: np.abs(x - y)
#pdb.set_trace()
#dist, cost, path = dtw(mfccs[0], chorus_mfcc, dist=euclidean_norm)


#
# chorus_times = [list(i) for i in chorus_times]
# merged_chorus_times = []
# merged_chorus_times.append(chorus_times[0])
# for curr in chorus_times:
#     prev = merged_chorus_times[-1]
#     if curr[0] <= prev[1]:
#         #pdb.set_trace()
#         prev[1] = max(prev[1], curr[1])
#     else:
#         merged_chorus_times.append(curr)
#
# print(merged_chorus_times)


# idx = 0
# for time in chorus_times:
#     print("chorus found at {0:g} min {1:.2f} sec".format(
#             time[0] // 60, time[0] % 60))
#     chorus_wave_data = song_wav_data[int(time[0]*sr) : int(time[1]*sr)]
#     sf.write("chorus_segment_{}.wav".format(idx), chorus_wave_data, sr)
#     idx += 1
#time_lag_similarity.display()
