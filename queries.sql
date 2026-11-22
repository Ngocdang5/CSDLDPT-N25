-- queries.sql : cac truy van SQL mau de minh hoa CSDL (chup man hinh cho bao cao Phan 3)

-- 1) So ban ghi cua cac bang
SELECT (SELECT COUNT(*) FROM audio)                 AS audio,
       (SELECT COUNT(*) FROM audio_feature)         AS audio_feature,
       (SELECT COUNT(*) FROM audio_feature_scaled)  AS audio_feature_scaled,
       (SELECT COUNT(*) FROM scaler_param)          AS scaler_param;

-- 2) Chia tap database / test
SELECT split, COUNT(*) AS so_file, ROUND(SUM(duration)::numeric / 60, 1) AS tong_phut
FROM audio GROUP BY split ORDER BY split;

-- 3) Thong ke tempo, RMS, centroid cua toan bo tap (vector goc)
SELECT ROUND(MIN(tempo)::numeric, 1) AS tempo_min, ROUND(AVG(tempo)::numeric, 1) AS tempo_tb,
       ROUND(MAX(tempo)::numeric, 1) AS tempo_max,
       ROUND(AVG(rms_std)::numeric, 4) AS rms_std_tb,
       ROUND(AVG(centroid_mean)::numeric, 1) AS centroid_tb_hz
FROM audio_feature;

-- 4) 10 file co tempo cao nhat (kiem tra gia tri cuc doan, co the la loi gap doi)
SELECT a.file_name, ROUND(f.tempo::numeric, 1) AS tempo
FROM audio a JOIN audio_feature f USING (audio_id)
ORDER BY f.tempo DESC LIMIT 10;

-- 5) File "am sang" va "nhanh": centroid va tempo cao hon trung binh
SELECT a.file_name, ROUND(f.centroid_mean::numeric, 0) AS centroid_hz, ROUND(f.tempo::numeric, 1) AS tempo
FROM audio a JOIN audio_feature f USING (audio_id)
WHERE f.centroid_mean > (SELECT AVG(centroid_mean) FROM audio_feature)
  AND f.tempo > (SELECT AVG(tempo) FROM audio_feature)
ORDER BY f.tempo DESC LIMIT 10;

-- 6) Tham so chuan hoa cua tung bo dac trung (bo A, B, C)
SELECT grp, COUNT(*) AS so_chieu, ROUND(AVG(std)::numeric, 3) AS std_tb, SUM(was_constant::int) AS chieu_hang_so
FROM scaler_param GROUP BY grp ORDER BY grp;

-- 7) Vector chuan hoa cua mot file (8 chieu dau cua bo A)
SELECT a.file_name, s.mfcc_1_mean, s.mfcc_1_std, s.mfcc_2_mean, s.mfcc_2_std,
       s.contrast_1_mean, s.centroid_mean, s.tempo
FROM audio a JOIN audio_feature_scaled s USING (audio_id) ORDER BY a.audio_id LIMIT 3;
