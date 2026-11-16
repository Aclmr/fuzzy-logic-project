"""
NIST Ground Array 1-dakikalik verisini makaledeki formata indirger.
Kullanim:  python prepare_nist.py <indirilen_klasor>  [cikti.csv]
Klasordeki tum .csv ve .zip dosyalarini okur, sadece gereken sutunlari alir,
saatlik ortalamaya cevirir, gece saatlerini atar -> nist_hourly.csv (~1-2 MB)
Gereken: pip install pandas
"""
import sys, re, zipfile, io, glob, os
import pandas as pd

PATTERNS = {                      # hedef sutun : NIST sutun adi regex'i
    "irradiance":  r"^RefCell1_Wm2",
    "amb_temp":    r"^AmbTemp_C",
    "module_temp": r"^RTD_C(_Avg)?\(?[1-9]\)?$|^RTD_C(_Avg)?_?[1-9]$",
    "wind_speed":  r"^WindSpeedAve_ms|^WindSpeed_ms",
    "dc_voltage":  r"^InvVDCin",
    "dc_current":  r"^InvIDCin",
    "dc_power_kw": r"^InvPDC_kW",
}

def pick(cols):
    out = {}
    for k, pat in PATTERNS.items():
        m = [c for c in cols if re.search(pat, c.strip(), re.I)]
        if m: out[k] = m if k == "module_temp" else [m[0]]
    tcol = next((c for c in cols if re.search(r"time|date", c, re.I)), None)
    return tcol, out

def read_one(fobj, name):
    head = pd.read_csv(fobj, nrows=0); fobj.seek(0)
    tcol, sel = pick(list(head.columns))
    if tcol is None or "irradiance" not in sel:
        print(f"  ATLANDI (sutunlar bulunamadi): {name}"); return None
    use = [tcol] + sorted({c for v in sel.values() for c in v})
    df = pd.read_csv(fobj, usecols=use, low_memory=False)
    res = pd.DataFrame({"time": pd.to_datetime(df[tcol], errors="coerce")})
    for k, cs in sel.items():
        res[k] = df[cs].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return res

frames = []
src = sys.argv[1] if len(sys.argv) > 1 else "."
for path in sorted(glob.glob(os.path.join(src, "**", "*"), recursive=True)):
    if path.lower().endswith(".csv"):
        print("okunuyor:", path)
        with open(path, "rb") as f: r = read_one(io.BytesIO(f.read()), path)
        if r is not None: frames.append(r)
    elif path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.lower().endswith(".csv"):
                    print("okunuyor:", path, "->", n)
                    r = read_one(io.BytesIO(z.read(n)), n)
                    if r is not None: frames.append(r)

if not frames: sys.exit("Hic uygun CSV bulunamadi. Klasor yolunu kontrol et.")
df = pd.concat(frames).dropna(subset=["time"]).set_index("time").sort_index()
df = df[~df.index.duplicated()]
print("ham satir:", len(df), "| bulunan sutunlar:", list(df.columns))
hourly = df.resample("1h").mean()
hourly["n_minutes"] = df["irradiance"].resample("1h").count()
hourly = hourly[(hourly["n_minutes"] >= 45) & (hourly["irradiance"] > 20)]
if "dc_power_kw" in hourly: hourly["dc_power_w"] = hourly.pop("dc_power_kw") * 1000
out = sys.argv[2] if len(sys.argv) > 2 else "nist_hourly.csv"
hourly.round(4).to_csv(out)
print(f"YAZILDI: {out}  ({len(hourly)} saatlik gunduz satiri)")
