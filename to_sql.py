import os
import sqlite3
import pandas as pd

CSV_FILENAME = "loan_data_new.csv"
DB_NAME = "loans.db"
TARGET_TABLE = "loan_records"

print("Step 1: Membaca file CSV lokal...")
if not os.path.exists(CSV_FILENAME):
    print(f"Error: File {CSV_FILENAME} tidak ditemukan!")
    exit()

df = pd.read_csv(CSV_FILENAME)

print("Step 2: Mentransformasi nama kolom...")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
if 'home_onwership' in df.columns:
    df.rename(columns={'home_onwership': 'home_ownership'}, inplace=True)

# ─── MODIFIKASI DIMULAI DARI SINI (PROSES ETL KONVERSI CURRENCY) ───
print("Step 3: Mengonversi satuan USD ke IDR (Kurs: 16.000)...")
KURS_USD_TO_IDR = 18000 #rate terbaru

# Pastikan kolom ada di data mentah sebelum dikalikan untuk menghindari error
if 'person_income' in df.columns and 'loan_amount' in df.columns:
    df['person_income'] = df['person_income'] * KURS_USD_TO_IDR
    df['loan_amount'] = df['loan_amount'] * KURS_USD_TO_IDR
else:
    print("Kolom 'person_income' atau 'loan_amount' tidak ditemukan untuk konversi.")
# ───────────────────────────────────────────────────────────────────

print("Step 4: Membuat ID Unik berformat NSB_00001...")
if 'nasabah_id' not in df.columns:
    df.insert(0, 'nasabah_id', [f"NSB_{i:05d}" for i in range(1, len(df) + 1)])

print("Step 5: Mengimpor ke SQLite3...")
conn = sqlite3.connect(DB_NAME)

# Buat tabel baru/timpa dengan kolom nasabah_id di paling kiri dan nilai IDR Rupiah
df.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
conn.close()

print(f"Sukses: {len(df):,} data berhasil dikonversi ke Rupiah dan masuk ke '{DB_NAME}' dengan ID berformat NSB_00001!")