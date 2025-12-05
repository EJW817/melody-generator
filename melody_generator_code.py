import urllib.request
import io
from IPython.display import display, Audio, HTML
from scipy.io import wavfile
import numpy as np
from datetime import datetime
import os
import traceback
import random

MAX_INT = 32768

settings = {
    "input_file": "bg_track.wav",
    "output_file": "output_motif_sections.wav",
    "SOUND_URL": "https://drive.google.com/uc?export=download&id=1mjrNw-LFh_Hjk4-ww98zrbPqyeX_Nv5-",
    "BPM": 100,
    "chord_progression": [
        "C", "F", "G", "C",
        "Am", "F", "G", "C",
        "Dm", "G", "C", "F",
        "C", "G", "Am", "F"
    ],
    "chord_notes": {
        "C": [60, 64, 67, 72],
        "F": [60, 65, 69, 72],
        "G": [62, 67, 71, 74],
        "Am": [57, 60, 64, 69],
        "Dm": [62, 65, 69, 74],
    },
    # First/second half pitch ranges (from reference)
    "first_half_range": [60, 62, 64, 65, 67, 71, 72, 74],
    "second_half_range": [67, 71, 72, 74, 76, 77, 79],
    # Chord-specific available notes for each half (from reference)
    "first_half_chords": {
        "C": [60, 62, 64, 65, 67, 71, 72],
        "Am": [60, 64],
        "F": [60, 64, 65, 67, 71, 72],
        "Dm": [62, 65],
        "G": [67, 71, 74],
    },
    "second_half_chords": {
        "C": [67, 71, 72, 74, 76, 77, 79],
        "Am": [72, 76],
        "F": [67, 71, 72, 76, 77, 79],
        "Dm": [67, 74, 77],
        "G": [67, 71, 74, 79],
    },
    # Special pairs used for final two notes of developed motif
    "special_note_pairs": [
        [65, 64],
        [65, 67],
        [71, 72],
    ],
    "scale": [60, 62, 64, 65, 67, 69, 71, 72],
}

starting_note_rules = {
    60: [60, 64, 67, 72],
    62: [60, 64],
    65: [60, 64, 67, 71, 72],
    71: [64, 67, 72],
}

# Duration and probability
DURATION_WEIGHTS = {
    1.0: 0.40,
    0.5: 0.30,
    2.0: 0.15,
    1.5: 0.08,
    0.25: 0.05,
    0.75: 0.02
}

# 4 sections
SECTION_NOTE_RANGES = [
    (6, 8),
    (8, 11),
    (11, 12),
    (8, 11),
]


def print_log(msg: str):
    """

    :param msg:
    :return:
    Print message and append to LOG.txt.
    """
    try:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line)
        with open('LOG.txt', 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def recreate_log_file():
    """

    :return:
    Clear previous LOG.txt.
    """
    try:
        if os.path.exists('LOG.txt'):
            os.remove('LOG.txt')
    except Exception:
        pass


def load_file_by_url(url):
    """
    :param url:
    :return:
    Load WAV file by URL.
    """
    try:
        file = urllib.request.urlopen(urllib.request.Request(url))
        sample_rate, s = wavfile.read(io.BytesIO(file.read()))
        return sample_rate, s.astype('float32') / MAX_INT
    except Exception:
        print_log("Except exception in get_chord_for_beat: {}".format(str(e)))
        print_log(traceback.format_exc())
        return None, None


def synth_note_to_track(track, sr, start_sample, dur_samples, pitch):
    """
    :param track:
    :param sr:
    :param start_sample:
    :param dur_samples:
    :param pitch:
    :return:
    Add a simple sawtooth-like synthesized note.
    """
    try:
        end_sample = min(start_sample + dur_samples, len(track))
        dur_samples = end_sample - start_sample
        if dur_samples <= 0:
            return

        t = np.arange(dur_samples) / sr
        f = 440.0 * (2 ** ((pitch - 69) / 12.0))
        theta = 2 * np.pi * f * t

        note = np.sin(theta)
        for n in range(2, 6):
            note += (1.0 / n) * np.sin(n * theta)

        note *= np.logspace(0, -2, dur_samples)
        track[start_sample:end_sample] += note
    except Exception as e:
        print_log("Except exception in synth_note_to_track: {}".format(str(e)))
        print_log(traceback.format_exc())


def get_quarter_duration(settings):
    '''Return quarter note duration.'''
    return 60.0 / settings['BPM']


def get_chord_for_beat(beat, settings):
    """
    :param beat:
    :param settings:
    :return:
    Return chord active at this beat
    """
    try:
        bars = len(settings['chord_progression'])
        bar = int(beat // 4)
        bar = max(0, min(bars - 1, bar))
        return settings['chord_progression'][bar]
    except Exception as e:
        print_log("Except exception in get_chord_for_beat: {}".format(str(e)))
        print_log(traceback.format_exc())


def get_weighted_duration(max_beats_remaining):
    try:
        possible = [d for d in DURATION_WEIGHTS if d <= max_beats_remaining + 1e-9]
        if not possible:
            return None
        weights = [DURATION_WEIGHTS[d] for d in possible]
        return random.choices(possible, weights=weights, k=1)[0]
    except Exception as e:
        print_log("Except exception in get_weighted_duration: {}".format(str(e)))
        print_log(traceback.format_exc())


def generate_rhythm_once(num_notes):
    """
    Generate a rhythm with a fixed number of notes using weighs
    """
    try:
        rhythm = []
        total = 0
        max_beats = 8  # try to hit 8 beats

        for _ in range(num_notes):
            left = max_beats - total
            if left <= 0:
                break
            d = get_weighted_duration(left)
            if not d:
                break
            rhythm.append(d)
            total += d
        return rhythm
    except Exception as e:
        print_log("Except exception in generate_rhythm_once: {}".format(str(e)))
        print_log(traceback.format_exc())


def generate_best_rhythm(num_notes, samples=2000):
    """
    get the best possible rythm closest to 8 beats given appropriate weights
    """
    try:
        best = None
        best_diff = float("inf")

        for _ in range(samples):
            r = generate_rhythm_once(num_notes)
            if len(r) != num_notes:
                continue  # keep same number of notes
            diff = abs(sum(r) - 8)
            if diff < best_diff:
                best_diff = diff
                best = r

        return best
    except Exception as e:
        print_log("Except exception in generate_best_rhythm: {}".format(str(e)))
        print_log(traceback.format_exc())


def get_chord_notes_current(absolute_bar, second_half_start_bar, settings):
    """
    check if we should use the second_half_chords
    :param absolute_bar:
    :param second_half_start_bar:
    :param settings:
    :return:
    """
    try:
        if absolute_bar >= second_half_start_bar:
            return settings['second_half_chords']
        else:
            return settings['first_half_chords']
    except Exception as e:
        print_log("Except exception in get_chord_notes_current: {}".format(str(e)))
        print_log(traceback.format_exc())


def apply_rules(pos, chord, last_note, notes, bar_idx, second_half_start_bar, current_range, repeat_count):
    """
    applyies rules to the notes
    :param pos:
    :param chord:
    :param last_note:
    :param notes:
    :param bar_idx:
    :param second_half_start_bar:
    :param current_range:
    :param repeat_count:
    :return:
    """
    try:
        filtered_notes = notes.copy()

        # Remove passing notes on even positions
        if 62 in filtered_notes and pos % 2 == 0:
            filtered_notes.remove(62)
        if 74 in filtered_notes and pos % 2 == 0:
            filtered_notes.remove(74)

        # Limit repetition
        if any(x >= 2 for x in [repeat_count.get(n, 0) for n in filtered_notes]):
            filtered_notes = [n for n in filtered_notes if repeat_count.get(n, 0) < 2] or filtered_notes

            if [n for n in filtered_notes if repeat_count.get(n, 0) < 2] != []:
                return [n for n in filtered_notes if repeat_count.get(n, 0) < 2]
            else:
                return filtered_notes
        else:
            filtered_notes = [n for n in filtered_notes if repeat_count.get(n, 0) < 2] or filtered_notes

        # Interval limit
        if last_note is not None:
            filtered_notes = [n for n in filtered_notes if abs(n - last_note) <= 12] or filtered_notes

        # Forbidden pairs
        if last_note is not None:
            forbidden_pairs = [(71, 62), (62, 71), (67, 77), (77, 67)]
            filtered_notes = [n for n in filtered_notes if (last_note, n) not in forbidden_pairs] or filtered_notes

        # B to C
        if last_note == 71:
            filtered_notes = [n for n in filtered_notes if n == 72] or [72]

        # First note second half
        if bar_idx > second_half_start_bar and pos == 0:
            allowed = [67, 71, 72]
            filtered_notes = [n for n in filtered_notes if n in allowed] or filtered_notes

        # Starting note C chord rule
        if chord == 'C' and pos % 4 == 0:
            new_set = []
            for sn, nxt in starting_note_rules.items():
                if sn in filtered_notes:
                    new_set.extend([x for x in nxt if x in filtered_notes])
            if new_set:
                filtered_notes = new_set

        # Fallback if empty
        if not filtered_notes:
            filtered_notes = notes.copy()

        return filtered_notes
    except Exception as e:
        print_log("Except exception in apply_rules: {}".format(str(e)))
        print_log(traceback.format_exc())


def has_three_in_a_row_test(lst):
    return any(lst[i] == lst[i + 1] == lst[i + 2] for i in range(len(lst) - 2))


def generate_motif(section_index, start_beat, settings):
    """
    generate single motif
    :param section_index:
    :param start_beat:
    :param settings:
    :return:
    """
    try:
        print_log("\n\nGeneration motif. Section: {}".format(section_index + 1))
        mn, mx = SECTION_NOTE_RANGES[section_index]
        num = random.randint(mn, mx)
        print_log("Num of notes: {}".format(num))
        rhythm = generate_best_rhythm(num)
        total_bars = len(settings['chord_progression'])
        second_half_start_bar = total_bars // 2
        motif = []
        last_note = None
        repeat_count = {}

        events = []
        pitches = []
        durations = []

        t = start_beat
        pos = 0
        for d in rhythm:
            beat = t
            absolute_bar = int(beat // 4)
            current_range = get_chord_notes_current(absolute_bar, second_half_start_bar, settings)
            total_bars = len(settings["chord_progression"])
            second_half_start_bar = total_bars // 2
            chord = get_chord_for_beat(beat, settings)

            # Prefer chord tones within the allowed range
            possible_notes = current_range[chord].copy()
            filtered_notes = apply_rules(pos, chord, last_note, possible_notes, absolute_bar,
                                         second_half_start_bar, current_range, repeat_count)
            note = random.choice(filtered_notes)
            motif.append(note)
            if last_note == note:
                repeat_count[note] = repeat_count.get(note, 0) + 3
            else:
                repeat_count[note] = 0
            last_note = note
            pitch = note
            events.append((t, d, pitch))
            pitches.append(pitch)
            durations.append(d)
            t += d
            pos += 1

        print("motif: {}".format(str(pitches)))
        if has_three_in_a_row_test(pitches):
            print("3 notes in a row detected!")
            print(pitches)
            input()
        print("motif rythm: {}".format(str(durations)))

        return events, rhythm
    except Exception as e:
        print_log("Except exception in generate_motif: {}".format(str(e)))
        print_log(traceback.format_exc())


def develop_motif(motif_events, start_beat, settings):
    """
    Develop motif by reusing its pitches but replacing the last two with a special pair
    :param motif_events:
    :param start_beat:
    :param settings:
    :return:
    """
    try:
        if not motif_events:
            return []

        # Extract original durations and pitches from the motif
        durations = [d for (_, d, _) in motif_events]
        pitches = [p for (_, _, p) in motif_events]
        # print(pitches)
        for pitch in pitches:
            # print(pitch)
            if not pitch in settings['scale']:
                pass
                # print("Note out of scale!: {}".format(pitch))

        # Create derived motif (copy), then apply special note pair to the final two notes
        derived_pitches = pitches.copy()
        if len(derived_pitches) >= 2 and "special_note_pairs" in settings:
            pair = random.choice(settings["special_note_pairs"])
            # Replace the last two pitches with the chosen pair
            derived_pitches[-2:] = pair
        print_log("Developed motif: {}".format(str(derived_pitches)))
        print_log("Developed motif rythm: {}".format(str(durations)))
        # Rebuild events at the new start_beat with the derived pitches and same rhythm
        events = []
        t = start_beat
        for d, p in zip(durations, derived_pitches):
            events.append((t, d, p))
            t += d
        return events
    except Exception as e:
        print_log("Except exception in develop_motif: {}".format(str(e)))
        print_log(traceback.format_exc())


def generate_all_motifs(settings):
    """
    Generate motifs for all 4 sections,
     then a derived motif using special note pairs.
    """
    try:
        events = []
        for sec in range(4):
            section_start = sec * 4 * 4
            dev_start = section_start + 8.0  # developed motif starts 2 bars later
            motif_events, rhythm = generate_motif(sec, section_start, settings)
            # Build derived motif from the original motif events
            dev_events = develop_motif(motif_events, dev_start, settings)
            events.extend(motif_events)
            events.extend(dev_events)
        return events
    except Exception as e:
        print_log("Except exception in generate_all_motifs: {}".format(str(e)))
        print_log(traceback.format_exc())


def generate_notes(sr, duration_sec, settings):
    """
    render all the motifs
    :param sr:
    :param duration_sec:
    :param settings:
    :return:
    """
    try:
        quarter = get_quarter_duration(settings)
        total_samples = int(duration_sec * sr)
        track = np.zeros(total_samples, dtype=np.float32)

        events = generate_all_motifs(settings)
        for start_beat, dur_beats, pitch in events:
            start_s = start_beat * quarter
            dur_s = dur_beats * quarter
            synth_note_to_track(track, sr, int(start_s * sr), int(dur_s * sr), pitch)
        return track
    except Exception as e:
        print_log("Except exception in generate_notes: {}".format(str(e)))
        print_log(traceback.format_exc())


def create_stereo_track(bg, notes):
    """

    :param bg:
    :param notes:
    :return:
    Left = background, Right = notes.
    """

    try:
        L = max(len(bg), len(notes))
        if len(bg) < L:
            bg = np.pad(bg, (0, L - len(bg)))
        if len(notes) < L:
            notes = np.pad(notes, (0, L - len(notes)))

        max_note = np.max(np.abs(notes)) or 1
        notes = notes / (2 * max_note)

        return np.stack((bg, notes), axis=-1)

    except Exception as e:
        print_log("Except exception in create_stereo_track: {}".format(str(e)))
        print_log(traceback.format_exc())


def save_and_play(bg_track, sample_rate, file_name):
    """
    # Make a sound player function that plays array "x" with a sample rate "rate", and labels it with "name"
    # Also write the same audio to the file "<name>.wav"
    :param bg_track:
    :param sample_rate:
    :param name:
    :return:
    """
    try:
        wavfile.write(file_name, sample_rate, bg_track)
        display(HTML(
            '<style> table, th, td {border: 0px; }</style> <table><tr><td>' + file_name +
            '</td><td>' + Audio(bg_track.T, rate=sample_rate)._repr_html_()[3:] + '</td></tr></table>'
        ))
    except Exception as e:
        print_log("Except exception in load_file_by_url: {}".format(str(e)))
        print_log(traceback.format_exc())


def create_stereo_track(bg_track, notes_track):
    """
    # Create a stereo track with bg_track on left and notes_track on right

    :param bg_track:
    :param notes_track:
    :return:
    """
    try:
        max_len = max(len(notes_track), len(bg_track))
        bg_track = np.append(bg_track, np.zeros(max_len - len(bg_track)))
        notes_track = np.append(notes_track, np.zeros(max_len - len(notes_track))) / (
                2 * np.max(notes_track)
        )
        stereo_buffer = np.stack((bg_track, notes_track), axis=-1)
        return stereo_buffer
    except Exception as e:
        print_log("Except exception in create_stereo_track: {}".format(str(e)))
        print_log(traceback.format_exc())


def main():
    try:
        recreate_log_file()
        sample_rate, bg_track = load_file_by_url(settings['SOUND_URL'])

        quarter = get_quarter_duration(settings)
        total_beats = len(settings['chord_progression']) * 4
        duration_sec = total_beats * quarter

        notes_track = generate_notes(sample_rate, duration_sec, settings)
        stereo_buffer = create_stereo_track(bg_track, notes_track)

        # Listen and save
        save_and_play(stereo_buffer, sample_rate, settings['output_file'])

        print_log("Done and saved!")
    except Exception as e:
        print_log(f"Exception in main: {e}")
        print_log(traceback.format_exc())


if __name__ == '__main__':
    main()
