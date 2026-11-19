import pandas as pd

df = pd.read_csv(r".\ket_qua\ket_qua_chi_tiet.csv", encoding="utf-8-sig")
s = df["rms_std"].dropna()

q1, q2 = s.quantile(1/3), s.quantile(2/3)
thap = (s <= q1).sum()
cao = (s >= q2).sum()
vua = len(s) - thap - cao

print(f"Số file: {len(s)}")
print(f"rms_std trung bình: {s.mean():.4f}")
print(f"rms_std trung vị: {s.median():.4f}")
print(f"Min - Max: {s.min():.4f} - {s.max():.4f}")
print(f"Ngưỡng thấp (<= {q1:.4f}): {thap} file ({thap/len(s)*100:.1f}%)  -> đều, liên tục")
print(f"Ngưỡng cao  (>= {q2:.4f}): {cao} file ({cao/len(s)*100:.1f}%)  -> nhiều chỗ nhấn")
print(f"Mức vừa: {vua} file")