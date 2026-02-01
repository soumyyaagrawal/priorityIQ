import sounddevice as sd
import numpy as np
import librosa
import time

# ---------------- CONFIG ----------------
SAMPLE_RATE = 22050
DURATION = 3

SILENCE_THRESHOLD = 0.005
CALM_VOLUME_MAX = 0.035
STRESS_VOLUME_MIN = 0.045

CALM_PITCH_STD_MAX = 35
STRESS_PITCH_STD_MIN = 60

# ---------------------------------------

def record_audio():
    print("\n🎙️ Microphone ACTIVE — listening for 3 seconds...\n")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("🎙️ Microphone OFF — analyzing audio...\n")
    return audio.flatten()

def analyze_audio(audio):
    # ---------- VOLUME ----------
    rms = np.sqrt(np.mean(audio ** 2))

    if rms < SILENCE_THRESHOLD:
        return result(
            "SILENT", rms, 0, 0,
            ["Silence or very low audio detected"]
        )

    # ---------- PITCH ----------
    pitches = librosa.yin(
        audio,
        fmin=80,
        fmax=500,
        sr=SAMPLE_RATE
    )
    pitches = pitches[~np.isnan(pitches)]

    avg_pitch = np.mean(pitches)
    pitch_std = np.std(pitches)

    reasons = []

    # ---------- SCREAMING ----------
    if rms > 0.08 and avg_pitch > 300:
        reasons.extend([
            "Extremely high volume",
            "Very high pitch (panic/scream)"
        ])
        return result("SCREAMING", rms, avg_pitch, pitch_std, reasons)

    # ---------- STRESSED ----------
    if rms >= STRESS_VOLUME_MIN and pitch_std >= STRESS_PITCH_STD_MIN:
        reasons.extend([
            "Raised voice energy",
            "High pitch instability"
        ])
        return result("STRESSED", rms, avg_pitch, pitch_std, reasons)

    # ---------- CALM ----------
    if rms <= CALM_VOLUME_MAX and pitch_std <= CALM_PITCH_STD_MAX:
        reasons.extend([
            "Normal speaking volume",
            "Stable pitch"
        ])
        return result("CALM", rms, avg_pitch, pitch_std, reasons)

    # ---------- FALLBACK ----------
    reasons.append("Moderate speech detected")
    return result("CALM", rms, avg_pitch, pitch_std, reasons)

def result(label, rms, pitch, pitch_std, reasons):
    return {
        "label": label,
        "volume_rms": round(float(rms), 4),
        "avg_pitch_hz": round(float(pitch), 1),
        "pitch_variation": round(float(pitch_std), 1),
        "reasons": reasons
    }

# ---------------- MAIN LOOP ----------------
if __name__ == "__main__":
    while True:
        audio = record_audio()
        r = analyze_audio(audio)

        print("🚨 ANALYSIS RESULT")
        print("------------------")
        print(f"Urgency Label     : {r['label']}")
        print(f"Volume (RMS)      : {r['volume_rms']}")
        print(f"Average Pitch Hz  : {r['avg_pitch_hz']}")
        print(f"Pitch Variation   : {r['pitch_variation']}")
        print("Reasons:")
        for reason in r["reasons"]:
            print(f" - {reason}")

        input("\nPress ENTER to analyze again or CTRL+C to exit\n")