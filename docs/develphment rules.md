# 🌟 Gemini Development Rules (as of July 23, 2025)

## 🔒 Rule 1: Do Not Modify Acquisition Protocols Without Permission
If you ask for changes to UI elements, naming, prompts, file formats, or graphs, I **must not** alter anything related to:
- Exposure timing
- Spectrometer calls (`ASQESpectrometer`)
- Acquisition logic
- Data capture loop or cycling behavior

Instead, I will confirm:  
> ⚠️ This change may affect acquisition. Do you want to proceed?

## 🧠 Rule 2: Canvas Is the Source of Truth During Development
While editing in the canvas:
- No fallback to earlier uploads or memory versions is allowed
- Canvas always represents the *most recent* and *official* working copy of the script

## 💾 Rule 3: Active Directory = Executable Ground Truth
If you run or modify a script in Thonny or your IDE and save it to disk:
- That file is more up-to-date than what’s in the canvas or memory
- If there is any confusion, I must ask:  
> “Should I load the version from your active directory?”

## 🔁 Rule 4: Daily Workflow for Version Stability
At the **end of each day**:
1. You zip your full `aseq-python` active directory
2. You upload it the **next morning**
3. I work only with that day's uploaded zip unless directed otherwise

## 🛑 Rule 5: No Consolidation or Rewriting Without Your Consent
I must **never restructure** or “clean up” your working code unless:
- You explicitly say “clean this up” or “optimize this block”
- I first ask:  
> “Do you want me to reformat or refactor this section?”

## 🔍 Rule 6: Compare All Modifications Line by Line
If you request changes:
- I must confirm the current canvas line count
- I must show the exact **line difference** between the original and modified script

## 📎 Rule 7: All Rule Updates Must Be Saved
When rules change:
- I’ll confirm:  
> “Do you want to update the rule file and include it in the next zip?”