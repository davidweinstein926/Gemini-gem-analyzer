# structural_peak_matcher.py

def detect_structural_features(flags):
    result = {
        "synthetic_cvd": False,
        "heat_treated": False,
        "type_ia_natural": False
    }
    if 'L' in flags:
        df = flags['L']
        if any("Raman Peak" in s and r == "Real Peak" for s, r in df['Classification'].items()) and \
           any("Raman Fluorescence" in s and r == "Real Peak" for s, r in df['Classification'].items()) and \
           any("CVD" in s and r in ["Real Peak", "Weak or Absent"] for s, r in df['Classification'].items()):
            result["synthetic_cvd"] = True
        if any("Treatment" in s and r == "Real Peak" for s, r in df['Classification'].items()):
            result["heat_treated"] = True
    if 'B' in flags:
        df = flags['B']
        if df.get("Is Significant") is False:
            result["type_ia_natural"] = True
    return result

def classify_structural_profile(flags):
    result = []
    if flags["synthetic_cvd"]:
        result.append("synthetic_cvd")
    if flags["heat_treated"]:
        result.append("heat_treated")
    if flags["type_ia_natural"]:
        result.append("type_ia_natural")
    return result

def interpret_structural_results(flags):
    structural_flags = detect_structural_features({
        'B': flags['bb'],
        'L': flags['laser'],
        'U': flags['uv']
    })
    profile = classify_structural_profile(structural_flags)
    if not profile:
        return "⚠️ No diagnostic structural features identified."
    labels = {
        "synthetic_cvd": "✅ Synthetic Diamond (CVD)",
        "heat_treated": "✅ Heat-Treated Natural Diamond",
        "type_ia_natural": "✅ Type Ia Natural Diamond"
    }
    return " + ".join([labels[p] for p in profile if p in labels])
