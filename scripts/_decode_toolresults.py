"""Decode browser tool-result spill files ({name,b64} JSON) into raw PDFs under Data/CBI_housing_raw/pdf."""
import json, base64, glob, os, re
TR = r"C:\Users\Sarah\.claude\projects\D--daghighe\3af440ca-007f-40c0-8abd-1c531b81dd37\tool-results"
OUT = os.path.join("Data", "CBI_housing_raw", "pdf"); os.makedirs(OUT, exist_ok=True)
DONE = os.path.join("Data", "CBI_housing_raw", ".decoded_spills.txt")
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
for f in sorted(glob.glob(os.path.join(TR, "mcp-Claude_Browser-javascript_tool-*.txt"))):
    if os.path.basename(f) in done: continue
    open(DONE, "a").write(os.path.basename(f) + chr(10))
    try:
        outer = json.load(open(f, encoding="utf8"))
        txt = outer[0]["text"] if isinstance(outer, list) else outer
        txt = txt.split("\n\n(captured at origin")[0].strip()
        obj = json.loads(txt)
        if isinstance(obj, str): obj = json.loads(obj)
        items = obj if isinstance(obj, list) else [obj]
        for it in items:
            if not isinstance(it, dict) or "b64" not in it: continue
            data = base64.b64decode(it["b64"])
            if it["name"].endswith(".pdf"):
                assert data[:5] == b"%PDF-", it["name"]; dest = OUT
            else:
                dest = os.path.join("Data", "CBI_economic_indicators_raw"); os.makedirs(dest, exist_ok=True)
            open(os.path.join(dest, it["name"]), "wb").write(data)
            print("saved", it["name"], len(data))
    except Exception as e:
        pass
