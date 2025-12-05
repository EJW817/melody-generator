🎼 Melody Generator (Python)

A rule-based melody generation system that composes new melodic lines over a provided backing track.
This project explores algorithmic composition through harmonic context, probabilistic rhythm modeling, and a simple motif → developed-motif structure.

🚀 Features

Generates melodic phrases over a chord progression

Rule-based pitch selection tied to harmonic context

Weighted-probability rhythm generator

Motif creation followed by a developed motif with controlled variation

Simple additive synthesis for audio rendering

Stereo output (left = backing track, right = generated melody)

Exports standard 16-bit PCM WAV compatible with all DAWs

🧩 How It Works
1. Harmonic Context

Each bar has an assigned chord. For every note event, the generator selects possible pitches from chord-specific note sets, then filters them with musical rules (interval limits, repetition control, forbidden transitions, etc.).

2. Rhythm Modeling

The rhythm engine searches thousands of candidate patterns and selects the one closest to an eight-beat phrase, ensuring predictable musical structure.

3. Motif → Developed Motif

The system first generates a motif that fits the local harmony.
It then builds a developed motif by reusing the motif’s material but reshaping the final notes to create tension–resolution behavior.

4. Audio Rendering

The melodic line is synthesized with a simple additive oscillator and merged with the background track in stereo.
Output is written as 16-bit WAV for easy use in DAWs or conversion to MP3.

📦 Installation
git clone https://github.com/yourusername/melody-generator
cd melody-generator
pip install -r requirements.txt

▶️ Usage

Place your backing track or URL in settings["SOUND_URL"] or replace the local file path.

Run:

python main.py


Output will appear as:

output_motif_sections.wav


Use any MP3 encoder (e.g., ffmpeg, lame) if you need compressed audio for web playback.

📁 Project Structure
.
├── main.py
├── generator/
│   ├── rhythm.py
│   ├── rules.py
│   ├── motif.py
│   └── synthesis.py
├── settings.json
├── LOG.txt
└── README.md


(Adjust to match your actual structure.)

🔬 Future Work

Add ML-based melodic modeling

Add interactive UI (e.g., Streamlit or a web interface)

Integrate MIDI export

Experiment with variable phrase lengths and contour models
