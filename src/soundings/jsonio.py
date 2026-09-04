"""Writing decibel figures into JSON, which has neither an infinity nor a not-a-number.

Python's own `json` writes `-Infinity` and `NaN` and reads them back without
complaint, so a record holding one round-trips here and passes every test. It is
not JSON: `jq`, `JSON.parse`, Go and Rust all refuse the file outright. The
archive is published for readers that are none of them, so a figure that cannot
be written has to be said in words instead.

**The three that occur here all mean something**, which is why they are not
simply dropped. A level difference is `-inf` when one setting produced no signal
at all -- the parameter silenced the part, which is the strongest finding the
comparison can make. What fails to repeat is `-inf` when the residual was
entirely accounted for by the noise floor, which is the *best* a setting can
score rather than a missing reading. And a residual is not-a-number when there
were not enough takes to form it. Writing `null` for all three says only that
there is no number, so the direction is written beside it.
"""

from __future__ import annotations

import math

BELOW = "below anything this chain can measure, rather than a figure it measured"
ABOVE = "above anything this chain can measure, rather than a figure it measured"
NOT_TAKEN = "not a number, so the figure was never formed rather than formed and found empty"

WHY_BEYOND = (
    "JSON has no infinity and no not-a-number, so a figure that ran off the end of what could be "
    "measured is written as null with its direction named here. The three that occur are a level "
    "difference where one setting produced no signal at all, a repeatability figure whose "
    "residual was entirely accounted for by the noise floor, and a residual there were not "
    "enough takes to form. None of them is a missing reading."
)


def db(value: float | None, places: int = 2) -> float | None:
    """A decibel figure as JSON: rounded when finite, null when it is not."""
    if value is None:
        return None
    number = float(value)
    return round(number, places) if math.isfinite(number) else None


def beyond(**named: float | None) -> dict[str, str]:
    """Which of these figures ran off the end, and in which direction.

    Empty for a record whose figures were all finite, so the key it fills is
    absent from such a record rather than present and empty.
    """
    out = {}
    for name, value in named.items():
        if value is None:
            continue
        number = float(value)
        if math.isnan(number):
            out[name] = NOT_TAKEN
        elif number == -math.inf:
            out[name] = BELOW
        elif number == math.inf:
            out[name] = ABOVE
    return out


def indexed(name: str, values) -> dict[str, str]:
    """The same, for a figure held as a list, naming the position that ran off."""
    return beyond(**{f"{name}[{i}]": v for i, v in enumerate(values)})


__all__ = ["ABOVE", "BELOW", "NOT_TAKEN", "WHY_BEYOND", "beyond", "db", "indexed"]
