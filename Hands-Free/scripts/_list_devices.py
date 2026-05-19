import sounddevice as sd

hosts = sd.query_hostapis()

print(f"Default devices (input, output): {sd.default.device}\n")

print("All input devices:")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] == 0:
        continue
    hname = hosts[d["hostapi"]]["name"]
    dname = d["name"]
    inch = d["max_input_channels"]
    sr = int(d["default_samplerate"])
    print(f"  {i}: [{hname}] {dname!r}  in={inch}  sr={sr}")

def _noop(indata, frames, t, status):
    pass

print("\nOpenability tests (no forced samplerate):")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] == 0:
        continue
    try:
        s = sd.InputStream(device=i, channels=1, blocksize=512, callback=_noop)
        s.start(); s.stop(); s.close()
        print(f"  {i}: OK")
    except Exception as e:
        print(f"  {i}: FAIL - {e}")
