from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_ident(name: str, *, kind: str = "identifier") -> str:
    if not _IDENT_RE.fullmatch(name):
        raise ValueError(f"Invalid {kind}: {name!r}")
    return name


class QuestDBQueryRunner(Protocol):
    def query_df(self, sql: str, params: Sequence[object] | None = None) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class FeatureQuery:
    sql: str
    params: tuple[object, ...] = ()
    postprocess: Callable[[pd.DataFrame], pd.DataFrame] | None = None

    def execute_df(self, store: QuestDBQueryRunner) -> pd.DataFrame:
        df = store.query_df(self.sql, self.params)
        if self.postprocess is None:
            return df
        return self.postprocess(df)
