import pdb
import numpy as np
import scipy.signal
import librosa
import soundfile as sf
from pychorus import create_chroma
from pychorus.similarity_matrix import TimeTimeSimilarityMatrix, TimeLagSimilarityMatrix, Line

# Denoising size in seconds
SMOOTHING_SIZE_SEC = 2.5

# Number of samples to consider in one chunk.
# Smaller values take more time, but are more accurate
N_FFT = 2**7

# For line detection
LINE_THRESHOLD = 0.15
MIN_LINES = 10
NUM_ITERATIONS = 10

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

clip_length = 10
chroma, song_wav_data, sr, song_length_sec = create_chroma("foo.wav")



time_time_similarity = TimeTimeSimilarityMatrix(chroma, sr)
time_lag_similarity = TimeLagSimilarityMatrix(chroma, sr)

num_samples = chroma.shape[1]

chroma_sr = num_samples / song_length_sec
smoothing_size_samples = int(SMOOTHING_SIZE_SEC * chroma_sr)
time_lag_similarity.denoise(time_time_similarity.matrix,
                            smoothing_size_samples)

clip_length_samples = clip_length * chroma_sr

candidate_rows = local_maxima_rows(time_lag_similarity.matrix)
lines = detect_lines(time_lag_similarity.matrix, candidate_rows,
                     clip_length_samples)

if len(lines) == 0:
    print("No choruses were detected. Try a smaller search duration")



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

chorus_times = [list(i) for i in chorus_times]
merged_chorus_times = []
merged_chorus_times.append(chorus_times[0])
for curr in chorus_times:
    prev = merged_chorus_times[-1]
    if curr[0] <= prev[1]:
        #pdb.set_trace()
        prev[1] = max(prev[1], curr[1])
    else:
        merged_chorus_times.append(curr)

print(merged_chorus_times)



idx = 0
for time in merged_chorus_times:
    print("chorus found at {0:g} min {1:.2f} sec".format(
            time[0] // 60, time[0] % 60))
    chorus_wave_data = song_wav_data[int(time[0]*sr) : int(time[1]*sr)]
    sf.write("repeated_segment_{}.wav".format(idx), chorus_wave_data, sr)
    idx += 1
#time_lag_similarity.display()
