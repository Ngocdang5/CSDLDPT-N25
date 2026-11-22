import pandas as pd

df = pd.read_csv(r".\ket_qua\ket_qua_chi_tiet.csv", encoding="utf-8-sig")
t = df["tempo_bpm_uoc_tinh"].dropna()
t = t[t > 0]

print(f"Số file có tempo: {len(t)}")
print(f"Min - Max: {t.min():.1f} - {t.max():.1f} BPM")
print(f"Trung bình: {t.mean():.1f} BPM")
print(f"Trung vị: {t.median():.1f} BPM")
print(f"Khoảng 25%-75%: {t.quantile(0.25):.1f} - {t.quantile(0.75):.1f} BPM")

bins = [0, 90, 120, 1000]
labels = ["Chậm (<90)", "Vừa (90-120)", "Nhanh (>120)"]
nhom = pd.cut(t, bins=bins, labels=labels, right=False)
pct = nhom.value_counts(normalize=True).reindex(labels) * 100
print("\nPhân bố theo nhóm tempo:")
for k, v in pct.items():
    print(f"  {k}: {v:.1f}%")