from __future__ import annotations

import numpy as np
import pandas as pd

def cvd(
    df: pd.DataFrame,
    *,
    open: str = "open",
    close: str = "close",
    volume: str = "volume",
    zero_base: bool = True,
    shift: int = 1,
) -> pd.Series:
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be >= 0")

    op = df[open].astype("float64")
    cl = df[close].astype("float64")
    vol = df[volume].astype("float64")

    sign = np.sign((cl - op).to_numpy(dtype="float64"))
    sign = np.nan_to_num(sign, nan=0.0)
    vd = vol.to_numpy(dtype="float64") * sign
    out = pd.Series(
        np.cumsum(vd, dtype="float64"),
        index=df.index,
        name="cvd",
        dtype="float64",
    )
    if shift:
        out = out.shift(shift)

    if zero_base:
        first_valid = out.dropna()
        if not first_valid.empty:
            out = out - float(first_valid.iloc[0])
    return out
