"""Column-level table comparison used for verification.

This is a port of the comparator in ELT-Bench's ``evaluation/eva_stage2.py``
(https://github.com/uiuc-kang-lab/ELT-Bench, CC BY-SA 4.0), kept behaviourally
identical so that rewards agree with the official metric:

* both tables are read as strings and re-sorted by the model's unique key, so
  the result does not depend on row order;
* every ground-truth column is matched case-insensitively against the
  candidate; extra candidate columns are ignored;
* text is compared case-insensitively after trimming;
* numbers use the "typed" rule: an integer column must be equal (float noise
  only), a fractional column must agree within 1% *and* within one unit of
  the last decimal place the two sides share.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

_INT_LITERAL = re.compile(r"[-+]?\d+(\.0*)?")


def as_string_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Round-trip a fetched frame through CSV and read it back as strings.

    The official evaluator compares the materialized CSV of both sides, which
    makes the comparison independent of driver-specific dtypes (``Decimal``,
    ``5`` vs ``5.0``). Doing the same here keeps the two in agreement.
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer, dtype=str, keep_default_na=True)


def _sort_series_key(col: pd.Series) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    n_non_null = col.notna().sum()
    if n_non_null > 0 and int(num.notna().sum()) == int(n_non_null):
        return num
    return col.astype("string").str.strip().str.lower()


def sort_by_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sort by ``keys`` (case-insensitive), all other columns as tie-breakers."""
    if df is None or df.empty:
        return df.reset_index(drop=True) if df is not None else df
    lower = {c.lower(): c for c in df.columns}
    resolved: list[str] = []
    for k in keys:
        actual = lower.get(k.lower())
        if actual is not None and actual not in resolved:
            resolved.append(actual)
    ordered_cols = resolved + [c for c in df.columns if c not in resolved]
    if not ordered_cols:
        return df.reset_index(drop=True)
    aux = pd.DataFrame({c: _sort_series_key(df[c]) for c in ordered_cols})
    order = aux.sort_values(by=ordered_cols, kind="mergesort", na_position="last").index
    return df.loc[order].reset_index(drop=True)


def _to_numeric_if_possible(s: pd.Series) -> pd.Series:
    coerced = pd.to_numeric(s, errors="coerce")
    if coerced.isna().sum() == s.isna().sum():
        return coerced
    return s


def _vectors_match(v1: pd.Series, v2: pd.Series, rtol: float = 1e-2, atol: float = 1e-9) -> bool:
    if len(v1) != len(v2):
        return False
    n1 = _to_numeric_if_possible(v1.reset_index(drop=True))
    n2 = _to_numeric_if_possible(v2.reset_index(drop=True))
    if pd.api.types.is_numeric_dtype(n1) and pd.api.types.is_numeric_dtype(n2):
        a = n1.to_numpy(dtype=float)
        b = n2.to_numpy(dtype=float)
        nan_match = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
        return bool(np.all(nan_match | close))
    for a, b in zip(v1, v2, strict=True):
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) or pd.isna(b):
            return False
        if isinstance(a, str) and isinstance(b, str):
            if a.strip().lower() != b.strip().lower():
                return False
        elif a != b:
            return False
    return True


def _decimals(v: str) -> int:
    try:
        e = Decimal(v.strip()).normalize().as_tuple().exponent
    except (InvalidOperation, ValueError):
        return 0
    return max(0, -e) if isinstance(e, int) else 0


def _column_decimals(s: pd.Series) -> int:
    """Most decimal places any value of the column carries (0 = integer column)."""
    s = s.dropna().astype(str).str.strip()
    if s.empty:
        return 0
    plain = s.str.fullmatch(r"[-+]?\d*\.?\d*")
    # Values with no "." have no fractional part and are dropped here.
    frac = s[plain].str.split(".", n=1).str[1].dropna().astype(str).str.rstrip("0").str.len()
    m = int(frac.max()) if len(frac) else 0
    for v in s[~plain].unique():  # scientific notation, 'inf', ...
        m = max(m, _decimals(v))
    return m


def _vectors_match_typed(
    gold: pd.Series,
    out: pd.Series,
    rtol: float = 1e-2,
    atol: float = 1e-9,
    noise: float = 1e-9,
    min_decimals: int = 1,
) -> bool:
    if len(gold) != len(out):
        return False
    gold = gold.reset_index(drop=True)
    out = out.reset_index(drop=True)
    n1, n2 = _to_numeric_if_possible(gold), _to_numeric_if_possible(out)
    if not (pd.api.types.is_numeric_dtype(n1) and pd.api.types.is_numeric_dtype(n2)):
        return _vectors_match(gold, out, rtol, atol)

    a, b = n1.to_numpy(dtype=float), n2.to_numpy(dtype=float)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if np.any(nan_a != nan_b):
        return False
    keep = ~nan_a
    a, b = a[keep], b[keep]
    if not np.all(np.isclose(a, b, rtol=rtol, atol=atol)):
        return False
    diff = np.abs(a - b)

    gold_decimals = _column_decimals(gold)
    if gold_decimals == 0:
        ok = (diff <= atol + noise * np.abs(b)) & (diff < 0.5)
        gs = gold[keep].astype(str).str.strip().to_numpy()
        os_ = out[keep].astype(str).str.strip().to_numpy()
        for i in np.where((np.abs(a) >= 2**53) | (np.abs(b) >= 2**53))[0]:
            if _INT_LITERAL.fullmatch(gs[i]) and _INT_LITERAL.fullmatch(os_[i]):
                ok[i] = Decimal(gs[i]) == Decimal(os_[i])
            elif _INT_LITERAL.fullmatch(gs[i]):
                ok[i] = a[i] == b[i]
            else:
                ok[i] = diff[i] <= atol + noise * abs(b[i])
        return bool(np.all(ok))

    places = max(min(gold_decimals, _column_decimals(out)), min_decimals)
    return bool(np.all(diff <= 10.0**-places * 1.000001 + atol + noise * np.abs(b)))


@dataclass
class ColumnComparison:
    """Outcome of comparing a candidate table against its reference."""

    matched: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)

    @property
    def match(self) -> bool:
        return not self.unmatched and not self.missed

    @property
    def total(self) -> int:
        return len(self.matched) + len(self.unmatched) + len(self.missed)


def compare_tables(
    gold: pd.DataFrame, candidate: pd.DataFrame, keys: list[str], rule: str = "typed"
) -> ColumnComparison:
    """Compare ``candidate`` against ``gold`` column by column.

    Both frames must hold strings (see :func:`as_string_frame`). They are
    sorted by ``keys`` first, then each gold column is compared with the
    same-named candidate column.
    """
    if rule not in ("typed", "rtol"):
        raise ValueError(f"unknown comparison rule: {rule}")
    vectors_match = _vectors_match_typed if rule == "typed" else _vectors_match
    gold = sort_by_keys(gold, keys)
    candidate = sort_by_keys(candidate, keys)
    result = ColumnComparison()
    columns_lower = {c.lower(): c for c in candidate.columns}
    for gold_col in gold.columns:
        actual = columns_lower.get(gold_col.lower())
        if actual is None:
            result.missed.append(gold_col)
        elif vectors_match(gold[gold_col], candidate[actual]):
            result.matched.append(gold_col)
        else:
            result.unmatched.append(gold_col)
    return result
