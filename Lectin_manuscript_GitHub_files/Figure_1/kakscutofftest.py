#!/usr/bin/env python3
"""
Confidence intervals for Ka/Ks (KaKs_Calculator-style table), and evidence that
the filter (Ka > 0.01, Ks < 1.2, codon length >= 200) removes low-Ks artefacts.

Input: tab/space/comma-separated table of ALL pairs (unfiltered) with columns
Seq_1 Seq_2 Ka Ks Ka_Ks EffectiveLen AverageS-sites AverageN-sites cN cS pN pS Note
Usage: python kaks_ci.py kaks.tsv     (omit file to run on simulated data)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------------- settings ----------------
KA_MIN, KS_MAX, LEN_MIN = 0.01, 1.2, 200   # LEN_MIN is in CODONS
EFFECTIVE_LEN_IS_NT = False                 # True: EffectiveLen is nucleotides (= S-sites + N-sites)
N_BOOT, N_PER_GENE, CONF, SEED = 10_000, 2_000, 0.95, 42
rng = np.random.default_rng(SEED)


# ---------------- load ----------------
def simulate(n=5000):
    L = rng.integers(60, 1500, n)                 # codons
    ks = rng.gamma(1.5, 0.4, n) * rng.choice([1, 0.02], n, p=[0.85, 0.15])
    ka = ks * rng.beta(2, 8, n)
    S = 3 * L * rng.uniform(.2, .3, n); N = 3 * L - S
    cS = rng.poisson(S * ks); cN = rng.poisson(N * ka)
    return pd.DataFrame({"Seq_1": range(n), "Seq_2": range(n), "Ka": ka, "Ks": ks,
                         "EffectiveLen": 3 * L, "AverageS-sites": S,
                         "AverageN-sites": N, "cN": cN, "cS": cS})


def read_table(path):
    """Read a .txt/.tsv/.csv. Tab first (KaKs_Calculator default), then whitespace."""
    for sep in ["\t", r"\s+", ","]:
        df = pd.read_csv(path, sep=sep, engine="python", index_col=False,
                         na_values=["NA", "nan", "-"])
        if {"Ka", "Ks", "cN", "cS"} <= set(df.columns):
            return df
    raise ValueError("Could not find columns Ka, Ks, cN, cS - check the header row.")


def load(path=None):
    df = simulate() if path is None else read_table(path)
    for c in ["Ka", "Ks", "EffectiveLen", "cN", "cS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")   # 'NA' strings -> NaN
    n0 = len(df)
    num = ["Ka", "Ks", "EffectiveLen", "cN", "cS"]
    n_inf = int(np.isinf(df[num]).any(axis=1).sum())
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)   # inf -> missing
    df = df.dropna(subset=num)
    n_ks0 = int((df["Ks"] <= 0).sum())
    df = df[df["Ks"] > 0].copy()
    print(f"Rows: {n0} | dropped: {n0 - len(df)} "
          f"(of which inf values: {n_inf}, Ks = 0 or negative: {n_ks0}, rest NA)")
    df["omega"] = df["Ka"] / df["Ks"]
    df["codons"] = df["EffectiveLen"] / 3 if EFFECTIVE_LEN_IS_NT else df["EffectiveLen"]
    df["pass"] = (df["Ka"] > KA_MIN) & (df["Ks"] < KS_MAX) & (df["codons"] >= LEN_MIN)
    return df


# ---------------- 1. bootstrap CI over genes ----------------
BIG_N = 5_000   # above this, use analytic CIs (BCa would need huge memory, and isn't needed)

def boot_ci(x, stat=np.median, method="BCa"):
    """CI for the median or mean of x.
    n <= BIG_N : BCa bootstrap.
    n  > BIG_N : median -> distribution-free CI from binomial order statistics;
                 mean   -> normal approximation (CLT). Both are instant and
                 accurate at this sample size."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return (np.nan, np.nan)
    if n > BIG_N:
        a = (1 - CONF) / 2
        if stat is np.median:
            xs = np.sort(x)
            lo_i = int(stats.binom.ppf(a, n, .5))
            hi_i = int(stats.binom.ppf(1 - a, n, .5))
            return xs[max(lo_i, 0)], xs[min(hi_i, n - 1)]
        z = stats.norm.ppf(1 - a)
        se = x.std(ddof=1) / np.sqrt(n)
        return x.mean() - z * se, x.mean() + z * se
    r = stats.bootstrap((x,), stat, n_resamples=N_BOOT, confidence_level=CONF,
                        method=method, random_state=rng, batch=200)
    return r.confidence_interval.low, r.confidence_interval.high


def summary_table(df):
    rows = []
    for label, d in [("All pairs", df), ("Retained", df[df["pass"]]),
                     ("Removed", df[~df["pass"]])]:
        w = d["omega"].values
        for name, f in [("median", np.median), ("mean", np.mean)]:
            lo, hi = boot_ci(w, f)
            rows.append(dict(set=label, n=len(w), statistic=name, estimate=f(w),
                             ci_low=lo, ci_high=hi, ci_width=hi - lo,
                             sd=np.std(w, ddof=1)))
    return pd.DataFrame(rows)


# ---------------- 2. spread of Ka/Ks as a function of Ks ----------------
def ci_by_ks_bin(df, n_bins=10):
    d = df.copy()
    d["ks_bin"] = pd.qcut(d["Ks"], n_bins, duplicates="drop")
    rows = []
    for b, g in d.groupby("ks_bin", observed=True):
        lo, hi = boot_ci(g["omega"].values, np.median)
        q1, q3 = g["omega"].quantile([.25, .75])
        rows.append(dict(ks_bin=str(b), ks_mid=g["Ks"].median(), n=len(g),
                         median_omega=g["omega"].median(), ci_low=lo, ci_high=hi,
                         ci_width=hi - lo, iqr=q3 - q1,
                         frac_omega_gt1=(g["omega"] > 1).mean()))
    return pd.DataFrame(rows)


# ---------------- 3. per-gene CI from the substitution counts ----------------
def per_gene_ci(df):
    """
    Parametric bootstrap per pair. The observed counts cN and cS are treated as
    Poisson means; each replicate redraws them and rescales the pair's own
    Ka and Ks (so whatever model produced them, e.g. YN/MYN/NG, is preserved):
        Ka* = Ka * cN*/cN ,  Ks* = Ks * cS*/cS ,  omega* = Ka*/Ks*
    Replicates with cS* = 0 give omega* = inf; a CI with an infinite upper
    bound means the data cannot rule out arbitrarily large Ka/Ks.
    """
    a = (1 - CONF) / 2
    n_rep = N_PER_GENE if len(df) <= 100_000 else 500   # keep runtime sane on big tables
    cN, cS = df["cN"].values, df["cS"].values
    ok = (cN > 0) & (cS > 0)
    lo = np.full(len(df), np.nan); hi = np.full(len(df), np.nan)
    for start in range(0, len(df), 2000):           # chunked to limit memory
        sl = slice(start, start + 2000)
        n, s, k = cN[sl], cS[sl], ok[sl]
        dn = rng.poisson(n[:, None], (len(n), n_rep))
        ds = rng.poisson(s[:, None], (len(s), n_rep))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (dn / n[:, None]) / (ds / s[:, None])
        ratio = np.where(ds == 0, np.inf, ratio)
        w = df["omega"].values[sl][:, None] * ratio
        lo[sl] = np.where(k, np.quantile(w, a, axis=1, method="lower"), np.nan)
        hi[sl] = np.where(k, np.quantile(w, 1 - a, axis=1, method="higher"), np.nan)
    out = df.copy()
    out["omega_lo"], out["omega_hi"] = lo, hi
    out["ci_fold_width"] = out["omega_hi"] / out["omega_lo"]   # inf if unbounded
    out["ci_excludes_1"] = (out["omega_hi"] < 1) | (out["omega_lo"] > 1)
    return out


# ---------------- plots ----------------
def make_plots(df, binned, g):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    sub = df.sample(min(len(df), 100_000), random_state=SEED)   # plot a subsample
    ax[0].scatter(sub["Ks"], sub["omega"], s=3, alpha=.3,
                  c=np.where(sub["pass"], "tab:blue", "tab:red"))
    ax[0].set(xscale="log", yscale="log", xlabel="Ks", ylabel="Ka/Ks",
              title="Ka/Ks vs Ks (red = removed by filter)")
    ax[0].axvline(KS_MAX, ls="--", c="k", lw=.8)

    ax[1].errorbar(binned["ks_mid"], binned["median_omega"],
                   yerr=[binned["median_omega"] - binned["ci_low"],
                         binned["ci_high"] - binned["median_omega"]],
                   fmt="o-", capsize=3)
    ax[1].set(xscale="log", xlabel="Ks (bin median)", ylabel="median Ka/Ks",
              title=f"{int(CONF*100)}% bootstrap CI by Ks bin")

    cap = 1e3
    for flag, c, lab in [(True, "tab:blue", "retained"), (False, "tab:red", "removed")]:
        fw = g.loc[(g["pass"] == flag) & g["ci_fold_width"].notna(), "ci_fold_width"]
        ax[2].hist(np.log10(np.clip(fw, 1, cap)), bins=50, alpha=.6, color=c,
                   label=lab, density=True)
    ax[2].set(xlabel="log10 fold-width of per-pair CI (capped at 3 = unbounded)",
              ylabel="density", title="Per-pair CI width")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig("kaks_ci_plots.png", dpi=200)


# ---------------- main ----------------
if __name__ == "__main__":
    df = load(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Pairs analysed: {len(df)} | retained: {df['pass'].sum()} | "
          f"removed: {(~df['pass']).sum()}")

    tab, binned, g = summary_table(df), ci_by_ks_bin(df), per_gene_ci(df)
    print("\nBootstrap CIs over pairs (BCa):\n", tab.round(4).to_string(index=False))
    print("\nBy Ks bin:\n", binned.round(4).to_string(index=False))

    for lab, m in [("retained", g["pass"]), ("removed", ~g["pass"])]:
        d = g[m & g["omega_lo"].notna()]
        print(f"\nPer-pair CIs, {lab} (n={len(d)}):"
              f"\n  median fold-width of CI : {d['ci_fold_width'].median():.2f}"
              f"\n  unbounded upper CI      : {np.isinf(d['omega_hi']).mean():.1%}"
              f"\n  CI excludes 1           : {d['ci_excludes_1'].mean():.1%}"
              f"\n  median cS               : {d['cS'].median():.1f}")

    tab.to_csv("kaks_bootstrap_summary.csv", index=False)
    binned.to_csv("kaks_ci_by_ks_bin.csv", index=False)
    g.to_csv("kaks_per_gene_ci.csv", index=False)
    make_plots(df, binned, g)
    print("\nSaved: kaks_bootstrap_summary.csv, kaks_ci_by_ks_bin.csv, "
          "kaks_per_gene_ci.csv, kaks_ci_plots.png")