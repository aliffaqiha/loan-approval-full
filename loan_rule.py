"""
Penyempurnaan Industri Perbankan:
    1. Mengunci DTI Maksimal Riil berdasarkan Segmentasi Pendapatan.
    2. Menghitung Dampak Suku Bunga terhadap Cicilan Bulanan secara matematis.
    3. Proteksi Mutlak (Hard Reject) terhadap over-indebtedness (gali lubang tutup lubang).
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class RuleResult:
    passed      : bool
    rule_name   : str
    severity    : str           # "HARD" = Tolak langsung | "WARN" = Potong confidence score model
    message     : str
    actual_value: Optional[str] = None
    threshold   : Optional[str] = None


# ══════════════════════════════════════════════════════
#  BANKING RISK CONFIGURATION
# ══════════════════════════════════════════════════════
RULES_CONFIG = {
    # ── HARD REJECT THRESHOLDS (Parameter Batas Aman Mutlak) ──
    "ABS_MAX_DTI_RATIO"       : 0.65,     # Tidak ada pinjaman lolos jika cicilan bulanan > 65% dari pendapatan
    "MIN_CREDIT_SCORE_HARD"   : 450,       # Di bawah 450 dikategorikan "Deep Subprime" (Auto-Reject)
    "MIN_AGE_HARD"            : 21,      # Standar legal perbankan untuk tanda tangan kontrak/akad (atau 17 jika menikah)
    "MAX_AGE_HARD"            : 65,       # Batas usia maksimal saat tenor pinjaman berakhir (Masa Pensiun)
    "MIN_INCOME_HARD"         : 12_000_000,# Rp 1 Juta/bulan adalah batas absolut kemiskinan ekstrem untuk kredit

    # ── WARNING & PENALTY THRESHOLDS (Sistem Poin Pengurang Bobot ML) ──
    "LOW_INCOME_TIER"         : 36_000_000, #Pendapatan < 3 Juta/bulan (Sangat sensitif terhadap fluktuasi ekonomi)
    "LOW_INCOME_MAX_DTI"      : 0.35,     # Batas DTI aman untuk Low Income (Maksimal 35%)
    "MID_INCOME_MAX_DTI"      : 0.50,     # Batas DTI aman untuk Middle-High Income (Maksimal 50%)
    
    "WARN_CREDIT_SCORE"       : 600,      # Skor < 600 masuk pengawasan khusus (Fair/Poor)
    "WARN_LOAN_TO_INCOME"     : 3.0,      # Total eksposur pinjaman tidak boleh > 3x pendapatan tahunan
    "WARN_NEW_EMPLOYEE"       : 1,       # Masa kerja < 1 tahun dianggap kurang stabil
}


def _fmt_idr(val: float) -> str:
    """Format angka ke Rupiah singkat."""
    if val >= 1_000_000_000: return f"Rp {val/1_000_000_000:.1f}M"
    if val >= 1_000_000:     return f"Rp {val/1_000_000:.1f} Jt"
    if val >= 1_000:         return f"Rp {val/1_000:.1f} Rb"
    return f"Rp {val:.0f}"


def check_all_rules(
    income       : float,  # Pendapatan tahunan (Rp)
    loan_amount  : float,   # Jumlah pinjaman (Rp)
    dti_ratio    : float,   # DTI kasar (dari simulasi dashboard)
    credit_score : int,    # Skor kredit (300-850)
    interest_rate: float,   # Suku bunga tahunan (%)
    credit_hist  : int,    # Riwayat kredit (Tahun)
    employee_exp : int,    # Pengalaman kerja (Tahun)
    age          : int,    # Usia (Tahun)
    prev_default : int,    # 1 = Pernah gagal bayar, 0 = Tidak
    loan_intent  : str = "",
) -> Dict[str, Any]:
    """
    Sistem verifikasi kelayakan berlapis menggunakan standar manajemen risiko perbankan.
    """
    hard_rejects = []
    warnings     = []
    passed       = []
    cfg          = RULES_CONFIG
    
    confidence_penalty = 0.0
    loan_to_income = loan_amount / income if income > 0 else float("inf")

    # ────────────────────────────────────────────────
    # RE-KALKULASI RASIO KEUANGAN BERDASARKAN SUKU BUNGA
    # ────────────────────────────────────────────────
    EST_TENOR_MONTHS = 36
    monthly_income = income / 12 if income > 0 else 0
    
    # Rumus PMT
    r_monthly = (interest_rate / 100) / 12
    if r_monthly > 0 and monthly_income > 0:
        est_monthly_installment = loan_amount * (r_monthly * (1 + r_monthly)**EST_TENOR_MONTHS) / ((1 + r_monthly)**EST_TENOR_MONTHS - 1)
        banking_dti = est_monthly_installment / monthly_income
    else:
        # Fallback ke rumus flat jika bunga 0 atau terjadi pembagian nol
        est_monthly_installment = (loan_amount * (1 + (interest_rate/100) * 3)) / EST_TENOR_MONTHS
        banking_dti = est_monthly_installment / (monthly_income if monthly_income > 0 else 1)

    # ────────────────────────────────────────────────
    # 1. HARD RULES (Penyaring Utama Kelayakan Finansial & Legal)
    # ────────────────────────────────────────────────

    # R1. Blacklist Gagal Bayar
    r = RuleResult(
        passed       = (prev_default == 0),
        rule_name    = "BI Checking / SLIK Blacklist",
        severity     = "HARD",
        message      = "Nasabah terdata memiliki riwayat gagal bayar aktif pada rekam jejak keuangan.",
        actual_value = "Kolektibilitas 5 (Macet)" if prev_default else "Kolektibilitas 1 (Lancar)",
        threshold    = "Wajib Bersih"
    )
    (hard_rejects if not r.passed else passed).append(r)

    # R2. Batas Maksimal DTI Absolut
    r = RuleResult(
        passed       = (banking_dti <= cfg["ABS_MAX_DTI_RATIO"]),
        rule_name    = "Batas Maksimal Beban Utang (DTI)",
        severity     = "HARD",
        message      = f"Beban cicilan bulanan mencapai {banking_dti*100:.1f}% dari total pendapatan resmi. Melanggar batas aman kapasitas bayar.",
        actual_value = f"{banking_dti*100:.1f}%",
        threshold    = f"≤ {cfg['ABS_MAX_DTI_RATIO']*100:.0f}%"
    )
    (hard_rejects if not r.passed else passed).append(r)

    # R3. Batas Minimum Pendapatan Hidup Layak
    r = RuleResult(
        passed       = (income >= cfg["MIN_INCOME_HARD"]),
        rule_name    = "Minimum Kelayakan Pendapatan",
        severity     = "HARD",
        message      = f"Pendapatan tahunan ({_fmt_idr(income)}) berada di bawah ambang batas dasar kredit produktif.",
        actual_value = _fmt_idr(income),
        threshold    = f"≥ {_fmt_idr(cfg['MIN_INCOME_HARD'])}"
    )
    (hard_rejects if not r.passed else passed).append(r)

    # R4. Batas Usia Kerja Produktif
    r = RuleResult(
        passed       = (cfg["MIN_AGE_HARD"] <= age <= cfg["MAX_AGE_HARD"]),
        rule_name    = "Kriteria Usia Produktif",
        severity     = "HARD",
        message      = f"Usia pemohon ({age} tahun) di luar rentang regulasi pembiayaan perbankan.",
        actual_value = f"{age} tahun",
        threshold    = f"{cfg['MIN_AGE_HARD']} - {cfg['MAX_AGE_HARD']} tahun"
    )
    (hard_rejects if not r.passed else passed).append(r)

    # R5. Deep Subprime Credit Score
    r = RuleResult(
        passed       = (credit_score >= cfg["MIN_CREDIT_SCORE_HARD"]),
        rule_name    = "Batas Minimum Skor Kredit",
        severity     = "HARD",
        message      = f"Skor Kredit ({credit_score}) terlalu rendah. Menunjukkan probabilitas gagal bayar jangka pendek yang ekstrem.",
        actual_value = str(credit_score),
        threshold    = f"≥ {cfg['MIN_CREDIT_SCORE_HARD']}"
    )
    (hard_rejects if not r.passed else passed).append(r)


    # ────────────────────────────────────────────────
    # 2. WARNING RULES (Penilaian Risiko Dinamis / Pemotongan Skor ML)
    # ────────────────────────────────────────────────

    # W1. Analisis DTI Berdasarkan Segmentasi Pendapatan (Rasionalisasi Kebijakan)
    if banking_dti <= cfg["ABS_MAX_DTI_RATIO"]:
        # Kasus A: Pendapatan Rendah (Low Income) -> Aturan DTI jauh lebih ketat
        if income < cfg["LOW_INCOME_TIER"]:
            is_passed = (banking_dti <= cfg["LOW_INCOME_MAX_DTI"])
            thold = cfg["LOW_INCOME_MAX_DTI"]
            msg = f"DTI Terlalu Tinggi untuk Kelas Pendapatan Rendah ({banking_dti*100:.1f}%). Sisa dana untuk kebutuhan hidup tidak memadai."
        # Kasus B: Pendapatan Menengah-Atas (Mid-High Income)
        else:
            is_passed = (banking_dti <= cfg["MID_INCOME_MAX_DTI"])
            thold = cfg["MID_INCOME_MAX_DTI"]
            msg = f"DTI Masuk Zona Pengawasan Risiko Tingkat Menengah ({banking_dti*100:.1f}%)."

        r = RuleResult(
            passed       = is_passed,
            rule_name    = "Segmented DTI Policy",
            severity     = "WARN",
            message      = msg,
            actual_value = f"{banking_dti*100:.1f}%",
            threshold    = f"≤ {thold*100:.0f}%"
        )
        if not r.passed:
            warnings.append(r)
            confidence_penalty += 0.25  # Penalti berat karena menyentuh porsi finansial bulanan
        else:
            passed.append(r)

    # W2. Eksposur Total Pinjaman Terhadap Pendapatan Bersih (Multiplier Loan-to-Income)
    r = RuleResult(
        passed       = (loan_to_income <= cfg["WARN_LOAN_TO_INCOME"]),
        rule_name    = "Capital Leverage Warning",
        severity     = "WARN",
        message      = f"Total plafon pinjaman melebihi {loan_to_income:.1f}x dari total pendapatan tahunan nasabah.",
        actual_value = f"{loan_to_income:.1f}x",
        threshold    = f"≤ {cfg['WARN_LOAN_TO_INCOME']}x"
    )
    if not r.passed:
        warnings.append(r)
        confidence_penalty += 0.15
    else:
        passed.append(r)

    # W3. Tingkat Kematangan Skor Kredit (Fair Category)
    r = RuleResult(
        passed       = (credit_score >= cfg["WARN_CREDIT_SCORE"]),
        rule_name    = "Credit Worthiness Warning",
        severity     = "WARN",
        message      = f"Skor Kredit masuk ke dalam area risiko menengah ({credit_score}). Tren pembayaran masa lalu berfluktuasi.",
        actual_value = str(credit_score),
        threshold    = f"≥ {cfg['WARN_CREDIT_SCORE']}"
    )
    if not r.passed:
        warnings.append(r)
        confidence_penalty += 0.10
    else:
        passed.append(r)

    # W4. Stabilitas Pekerjaan Pemohon
    r = RuleResult(
        passed       = (employee_exp >= cfg["WARN_NEW_EMPLOYEE"]),
        rule_name    = "Employment Stability Warning",
        severity     = "WARN",
        message      = f"Masa kerja kurang dari {cfg['WARN_NEW_EMPLOYEE']} tahun memiliki risiko gagal bayar akibat ketidakpastian arus pendapatan.",
        actual_value = f"{employee_exp} Tahun",
        threshold    = f"≥ {cfg['WARN_NEW_EMPLOYEE']} Tahun"
    )
    if not r.passed:
        warnings.append(r)
        confidence_penalty += 0.05
    else:
        passed.append(r)


    # ────────────────────────────────────────────────
    # 3. KEPUTUSAN FINAL
    # ────────────────────────────────────────────────
    if hard_rejects:
        final_decision = "HARD_REJECT"
        reject_reason = hard_rejects[0].message
    elif warnings:
        final_decision = "WARN"
        reject_reason = None
    else:
        final_decision = "PASS"
        reject_reason = None

    return {
        "final_decision"    : final_decision,
        "hard_rejects"      : hard_rejects,
        "warnings"          : warnings,
        "passed"            : passed,
        "confidence_penalty": min(confidence_penalty, 0.50),  # Pembatasan akumulasi penalti maksimal 50%
        "reject_reason"     : reject_reason,
        "banking_dti"       : banking_dti,                   
    }