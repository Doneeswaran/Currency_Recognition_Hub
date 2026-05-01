import struct, wave, math, os

os.makedirs('audio_files', exist_ok=True)

sample_rate = 22050
freq = 880.0  # High-pitched beep to indicate unknown

samples = []

# 3 short beeps = "Unknown detected" signal
beep_samples = int(sample_rate * 0.3)
silence_samples = int(sample_rate * 0.15)

for _ in range(3):
    for i in range(beep_samples):
        v = int(20000 * math.sin(2 * math.pi * freq * i / sample_rate))
        samples.append(struct.pack('<h', v))
    for i in range(silence_samples):
        samples.append(struct.pack('<h', 0))

with wave.open('audio_files/unknown.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    for s in samples:
        f.writeframesraw(s)

print("Created audio_files/unknown.wav successfully!")
