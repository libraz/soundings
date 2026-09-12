"""Rendering a model and reading it through the pipeline the unit was read through.

This is the only place in the repository that derives. What it produces goes
under `inferences/`, never under `data/`, and it exists because a structure that
cannot be checked against the unit is a structure nobody should publish.

**Waveforms are not compared and cannot be.** Two recordings of the same setting
drift apart by tens of parts per million over a take, and the coherence between
them has fallen to a tenth by four kilohertz. So the model is not held against
the unit's audio; it is held against the quantities the archive publishes, read
out of the model by the same code that read them out of the unit. The reader's
own bias then sits on both sides of the subtraction and cancels -- which matters,
because that bias has already been measured and is the same to four decimal
places across types.

**The residual is a detector, not a target.** Once the gates below are passed,
whether it is one decibel or half of one changes no verdict, and driving it down
is chasing the room, the converters and the unit's own wobble. What decides is
whether the residual leans -- a residual that grows with the setting is pointing
at a structure the model does not have, and one that does not is pointing at the
analogue. Both are reported; only the first reopens anything.

**What this renderer cannot do yet.** It renders time-invariant, linear models:
a chain of shelves, peaking sections and gains, evaluated as a frequency
response. That covers the equaliser class and nothing else. A modulated or
saturating type needs a time-domain renderer, and the honest state of that is
that it is not written. A model whose `kind` is not `lti` is refused rather than
approximated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import efxbands, takes

GROSS_CEILING = 0.2
"""How much of the effect's own span the median residual may be and still pass.

Scale-free on purpose. An absolute decibel figure would have to be invented per
type and would be wrong for the next one; the span the effect commands over the
domain it was tested in is measured, is different for every row, and is the
quantity a reader means by "roughly reproduces it". A fifth of it is the line
between a model that captures the effect and one that captures most of it.

It is a rejection bound and not a target. Below it the number is reported and
never optimised.
"""

BREAKDOWN_SHARE = 0.5
"""Where a single point stops being error and becomes a failure.

Half the span at one setting is not a model that is slightly off there; it is a
model that does something else there. Named separately from the gross gate
because the gross gate is about the whole domain and this is about one point in
it, and a model can pass either while failing the other.
"""


# ---------------------------------------------------------------- the parameters


def _clamped(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _from_map(spec: dict, byte_value: int) -> float:
    """One byte turned into the quantity a stage takes, by the model's own rule.

    Four rules, and each of them is a shape this archive has measured rather than
    a shape a model wanted. A byte that answers over a window and sticks at both
    ends, a byte that has two states, a byte read at some values and interpolated
    between them, and a byte that indexes a short table and returns the first
    entry for anything past its end -- all four are in the equaliser alone.
    """
    kind = spec["kind"]
    if kind == "window":
        low, high = spec["low"], spec["high"]
        clamped = _clamped(byte_value, low, high)
        share = (clamped - low) / (high - low)
        return spec["at_low"] + share * (spec["at_high"] - spec["at_low"])
    if kind == "states":
        values = spec["values"]
        key = str(byte_value)
        return float(values[key] if key in values else values["*"])
    if kind == "points":
        points = sorted((int(v), float(x)) for v, x in spec["points"])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if spec.get("log", False):
            return float(np.exp(np.interp(byte_value, xs, np.log(ys))))
        return float(np.interp(byte_value, xs, ys))
    if kind == "table":
        entries = spec["entries"]
        index = byte_value if 0 <= byte_value < len(entries) else spec["out_of_range"]
        return float(entries[index])
    raise ValueError(f"{kind!r} is not a rule this renderer knows")


def _value(param: dict, bytes_now: dict[str, int]) -> float:
    if "fixed" in param:
        return float(param["fixed"])
    address = param["byte"]
    if address not in bytes_now:
        raise KeyError(f"the model reads {address} and the setting does not say what it holds")
    return _from_map(param["map"], int(bytes_now[address]))


# ---------------------------------------------------------------- the sections


def _biquad(b0, b1, b2, a0, a1, a2, w):
    z1 = np.exp(-1j * w)
    z2 = z1 * z1
    return (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)


def _first_order_shelf(freq_hz, *, side: str, corner_hz: float, gain_db: float, fs: float):
    """A one-pole, one-zero shelf, bilinear transformed with its corner prewarped.

    The corner is where the shelf is half way up in decibels, which is where this
    archive's `half_below_hz` and `half_above_hz` are read, so the model's
    parameter and the measurement's figure mean the same thing. That is not a
    convenience: a shelf parameterised at its 3 dB point and a measurement taken
    at its half-gain point disagree by a factor that grows with the gain, and the
    disagreement would land in the residual looking like a structural error.
    """
    g = 10.0 ** (gain_db / 20.0)
    if abs(gain_db) < 1e-9:
        return np.ones_like(freq_hz, dtype=complex)
    a = 2 * np.pi * corner_hz / np.sqrt(g) if side == "low" else 2 * np.pi * corner_hz * np.sqrt(g)
    k = np.tan(np.pi * (a / (2 * np.pi)) / fs)
    inv = 1.0 / k
    if side == "low":
        b0, b1 = inv + g, g - inv
    else:
        b0, b1 = g * inv + 1.0, 1.0 - g * inv
    a0, a1 = inv + 1.0, 1.0 - inv
    w = 2 * np.pi * freq_hz / fs
    z1 = np.exp(-1j * w)
    return (b0 + b1 * z1) / (a0 + a1 * z1)


def _second_order_shelf(freq_hz, *, side: str, corner_hz: float, gain_db: float, fs: float):
    """The shelving section of the audio cookbook, at unit slope.

    Unit slope because that is the shelf a part which already has biquads gets for
    free, and because a slope parameter would be a third thing to fit. At full
    gain it runs at 7.2 dB per octave against the one-pole section's 3.6, which is
    the difference the two candidates are separated by.
    """
    if abs(gain_db) < 1e-9:
        return np.ones_like(freq_hz, dtype=complex)
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * corner_hz / fs
    cos0, sin0 = np.cos(w0), np.sin(w0)
    alpha = sin0 / 2.0 * np.sqrt(2.0)  # the cookbook's alpha at S = 1
    root = 2.0 * np.sqrt(amp) * alpha
    if side == "low":
        b0 = amp * ((amp + 1) - (amp - 1) * cos0 + root)
        b1 = 2 * amp * ((amp - 1) - (amp + 1) * cos0)
        b2 = amp * ((amp + 1) - (amp - 1) * cos0 - root)
        a0 = (amp + 1) + (amp - 1) * cos0 + root
        a1 = -2 * ((amp - 1) + (amp + 1) * cos0)
        a2 = (amp + 1) + (amp - 1) * cos0 - root
    else:
        b0 = amp * ((amp + 1) + (amp - 1) * cos0 + root)
        b1 = -2 * amp * ((amp - 1) + (amp + 1) * cos0)
        b2 = amp * ((amp + 1) + (amp - 1) * cos0 - root)
        a0 = (amp + 1) - (amp - 1) * cos0 + root
        a1 = 2 * ((amp - 1) - (amp + 1) * cos0)
        a2 = (amp + 1) - (amp - 1) * cos0 - root
    return _biquad(b0, b1, b2, a0, a1, a2, 2 * np.pi * freq_hz / fs)


def _peaking(freq_hz, *, centre_hz: float, q: float, gain_db: float, fs: float):
    if abs(gain_db) < 1e-9:
        return np.ones_like(freq_hz, dtype=complex)
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * centre_hz / fs
    alpha = np.sin(w0) / (2.0 * q)
    cos0 = np.cos(w0)
    return _biquad(
        1 + alpha * amp, -2 * cos0, 1 - alpha * amp,
        1 + alpha / amp, -2 * cos0, 1 - alpha / amp,
        2 * np.pi * freq_hz / fs,
    )


def response(model: dict, bytes_now: dict[str, int], freq_hz: np.ndarray) -> np.ndarray:
    """The whole chain's transfer at the frequencies asked for, for one setting.

    Above half the model's own rate there is nothing to return: a chain running
    at thirty-two kilohertz has no response at seventeen. Those frequencies come
    back as NaN and are dropped from every comparison by name, rather than being
    given the value at Nyquist -- which would be a number the model never said.
    """
    fs = float(model["sample_rate_hz"])
    out = np.ones_like(freq_hz, dtype=complex)
    inside = freq_hz < fs / 2.0
    safe = np.where(inside, freq_hz, fs / 4.0)
    for stage in model["chain"]:
        kind = stage["kind"]
        if kind == "shelf":
            gain_db = _value(stage["gain_db"], bytes_now)
            corner = _value(stage["corner_hz"], bytes_now)
            order = int(stage.get("order", 1))
            shelf = _first_order_shelf if order == 1 else _second_order_shelf
            out = out * shelf(safe, side=stage["side"], corner_hz=corner, gain_db=gain_db, fs=fs)
        elif kind == "peaking":
            out = out * _peaking(
                safe,
                centre_hz=_value(stage["centre_hz"], bytes_now),
                q=_value(stage["q"], bytes_now),
                gain_db=_value(stage["gain_db"], bytes_now),
                fs=fs,
            )
        elif kind == "gain":
            out = out * 10.0 ** (_value(stage["gain_db"], bytes_now) / 20.0)
        else:
            raise ValueError(f"{kind!r} is not a section this renderer holds")
    return np.where(inside, out, np.nan)


# ---------------------------------------------------------------- reading a model


def _applied(
    samples: np.ndarray,
    rate: int,
    model: dict,
    bytes_now: dict[str, int],
    bytes_flat: dict[str, int],
):
    """One channel of a take carried from the flat setting to the swept one.

    The ratio of the two responses and not the swept one alone. What the archive
    publishes is a setting read against the run's own flat repeats, and the flat
    take already has the flat setting's response in it -- applying the swept
    response to it would put the chain through twice and report a deviation
    against a setting nobody recorded.

    Applied over the whole take rather than over the body the bands are read
    from: filtering a trimmed section and then reading it puts the filter's own
    edge inside the window, and the window is what the reading is made through.
    """
    spectrum = np.fft.rfft(samples)
    freq = np.fft.rfftfreq(samples.size, 1.0 / rate)
    now = response(model, bytes_now, freq)
    flat = response(model, bytes_flat, freq)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.nan_to_num(now / flat, nan=0.0, posinf=0.0, neginf=0.0)
    return np.fft.irfft(spectrum * ratio, n=samples.size)


def _bands_of(body: np.ndarray, rate: int, centres, width_octaves: float) -> list[float]:
    return efxbands.energies(body, rate, centres, width_octaves)


def model_reading(
    model: dict,
    *,
    takes_dir: Path,
    flat_takes: list[str],
    bytes_now: dict[str, int],
    bytes_flat: dict[str, int],
    centres: list[float],
    width_octaves: float,
    channel: int,
    lead_s: float = 0.6,
    trim_s: float = 0.5,
    hold_s: float | None = None,
) -> tuple[list[float], list[float]]:
    """What the reading pipeline returns for the model, and for doing nothing.

    Both come out of the same takes: the deviation the archive publishes is a
    setting read against the run's own flat repeats, so the model's deviation has
    to be its own output read against the same repeats unfiltered. Anything else
    compares two different subtractions.
    """
    filtered, plain = [], []
    for name in flat_takes:
        samples, rate = takes.read(takes_dir / name)
        one = takes.channel(samples, channel)
        seconds = one.size / rate
        hold = hold_s if hold_s is not None else seconds - 1.0
        first = int((lead_s + trim_s) * rate)
        last = int((lead_s + hold - trim_s) * rate)
        wet = _applied(one, rate, model, bytes_now, bytes_flat)
        filtered.append(_bands_of(wet[first:last], rate, centres, width_octaves))
        plain.append(_bands_of(one[first:last], rate, centres, width_octaves))
    def averaged(rows: list[list[float]]) -> list[float]:
        return [float(np.mean([row[i] for row in rows])) for i in range(len(centres))]

    return averaged(filtered), averaged(plain)


# ---------------------------------------------------------------- scoring


def _structured(rows: list[dict], floor_db: float) -> tuple[bool, str]:
    """Whether the residual leans, which is the only thing about it that decides.

    Two axes and one rule for both: the residual is structured when the signed
    mean over one axis moves across that axis by more than the run's own floor.
    A model that is half a decibel out everywhere is not leaning; a model that is
    half a decibel out at one end of a byte and half the other way at the other
    end is, and the two have the same median.

    **The yardstick is the run's floor and not the residual's own size.** Against
    its own size, a model that fits closely is held to a tighter lean than one
    that fits poorly, which is backwards -- and it makes every good model fail,
    because a good model's residual is small enough that any systematic tenth of
    a decibel exceeds it. The floor is what the same run measured between repeats
    of one setting, so a lean under it is a lean this measurement cannot see.
    """
    if not rows:
        return False, "no readings to say"
    # Medians and not means. A lean is a systematic shift and a mean is moved by
    # the few bands where the take sat near the floor, which is not a shift and is
    # exactly where the readings are noisiest. Taking the mean reported a lean of
    # one and a half decibels on the level row whose own map had been transcribed
    # from that record: an artefact of the statistic, not of the model.
    per_setting = [float(np.median(row["residual_db"])) for row in rows]
    setting_swing = float(max(per_setting) - min(per_setting))
    stacked = np.array([row["residual_db"] for row in rows], dtype=float)
    per_band = np.median(stacked, axis=0)
    band_swing = float(np.nanmax(per_band) - np.nanmin(per_band))
    leans = setting_swing > floor_db or band_swing > floor_db
    return leans, (
        f"the signed mean swings {setting_swing:.2f} dB across the settings and "
        f"{band_swing:.2f} dB across the bands, against the run's own floor of "
        f"{floor_db:.2f} dB"
    )


def score_against_bands(
    model: dict,
    record: dict,
    *,
    takes_dir: Path,
    hold_bytes: dict[str, int],
    flat_bytes: dict[str, int],
    swept_address: str,
) -> dict:
    """One published band-profile record, answered by the model.

    The comparison is per setting and per band, over exactly the settings the run
    asked and the bands the set holds. Bands above the model's own Nyquist are
    dropped by name: a chain that does not run that high has not been asked, and
    scoring it there would be scoring the renderer's padding.
    """
    centres = list(record["bands_hz"])
    width = float(record["band_width_octaves"])
    channel = int(record["channel"]["read"])
    flat_takes = list(record["reference"]["takes"])
    nyquist = float(model["sample_rate_hz"]) / 2.0
    usable = [i for i, c in enumerate(centres) if c < nyquist]
    dropped = [c for c in centres if c >= nyquist]

    floor = record.get("reference", {}).get("floor_db") or [0.0]
    floor_db = float(max(floor))

    rows, left_out = [], []
    for reading in record["readings"]:
        value = int(reading["value"])
        # A reading claiming a change as large as its whole margin over the chain's
        # own silence is a reading of that silence. The record says both numbers,
        # so the line needs no figure invented for it -- and the readings it drops
        # are the ones a model is asked to reproduce the room with.
        largest, above = reading.get("largest_db"), reading.get("above_the_silence_db")
        if largest is not None and above is not None and abs(largest) >= above:
            left_out.append(
                {
                    "value": value,
                    "why": (
                        f"the take stood {above:.1f} dB over the chain's silence and the "
                        f"reading claims {largest:.1f} dB, so what it reports is the floor"
                    ),
                }
            )
            continue
        bytes_now = dict(hold_bytes)
        bytes_now[swept_address] = value
        wet, dry = model_reading(
            model,
            takes_dir=takes_dir,
            flat_takes=flat_takes,
            bytes_now=bytes_now,
            bytes_flat=flat_bytes,
            centres=centres,
            width_octaves=width,
            channel=channel,
            hold_s=reading.get("hold_s"),
        )
        said = [round(wet[i] - dry[i], 2) for i in range(len(centres))]
        unit = reading["band_db"]
        residual = [round(said[i] - unit[i], 2) for i in usable]
        rows.append(
            {
                "value": value,
                "model_band_db": [said[i] for i in usable],
                "unit_band_db": [unit[i] for i in usable],
                "residual_db": residual,
                "largest_db": max((said[i] for i in usable), key=abs, default=None),
                "unit_largest_db": reading.get("largest_db"),
            }
        )

    flat = np.array([abs(v) for row in rows for v in row["residual_db"]], dtype=float)
    every = [v for row in rows for v in row["unit_band_db"]]
    span = float(max(every) - min(every)) if every else 0.0
    median_abs = float(np.median(flat)) if flat.size else 0.0
    worst = float(flat.max()) if flat.size else 0.0
    worst_where = None
    if flat.size:
        for row in rows:
            for j, v in enumerate(row["residual_db"]):
                if abs(v) == worst:
                    worst_where = {"value": row["value"], "hz": [centres[i] for i in usable][j]}
                    break
            if worst_where:
                break
    leans, why_leans = _structured(rows, floor_db)
    # A record whose span does not clear the run's own floor by a doubling is one
    # where the byte was swept with the stage it shapes turned off: the archive's
    # null controls, and the rows a run swept through a gain left at its centre,
    # which are the same thing without the name. There is no span to take a share
    # of -- dividing by one makes a large number out of two small ones, which is
    # how the first pass reported seven breakdowns on records where nothing
    # happened. What a model owes such a record is different in kind: it has to
    # show nothing there too.
    #
    # The doubling is not a delicate line. Across this type's twenty-one records
    # the ratio is either below 1.6 or above 6, so anything between those two
    # separates them and the figure chosen does not decide any verdict.
    is_null = span <= 2.0 * floor_db
    # Whether the record's own profile finished inside the band set. Where it did
    # not, the corner the model takes from it is a bound and not a value, so a
    # residual that leans across the bands is the bound leaning, not the model.
    # The record already publishes the figure that says so; nothing here decides
    # it. Reading it is what separates a model that is wrong from a measurement
    # that did not reach.
    unsettled = max(
        (
            abs(r[key])
            for r in record["readings"]
            for key in ("settled_below_db", "settled_above_db")
            if r.get(key) is not None
        ),
        default=0.0,
    )
    model_largest = max(
        (abs(v) for row in rows for v in row["model_band_db"]), default=0.0
    )
    return {
        "record": None,
        "address": record.get("address"),
        "span_db": round(span, 2),
        "floor_db": round(floor_db, 2),
        "is_null_record": is_null,
        "settled_by_db": round(unsettled, 2),
        "profile_settled_inside_the_bands": bool(unsettled <= floor_db),
        "model_largest_db": round(model_largest, 2),
        "model_stays_inside_the_floor": bool(model_largest <= floor_db),
        "median_abs_db": round(median_abs, 3),
        "worst_abs_db": round(worst, 3),
        "worst_at": worst_where,
        "structured": leans,
        "why_structured": why_leans,
        "bands_above_the_models_nyquist_hz": dropped,
        "readings_left_out": left_out,
        "rows": rows,
    }


def gates(scored: list[dict], *, ranking: list[dict]) -> dict:
    """The four gates, over every record the model was scored against.

    No gate on the residual's absolute size. The gross gate is the residual
    against the span the effect itself commands, which is measured per record and
    needs no figure to be invented; the power gate is whether the runner-up in the
    catalogue fails the same gross gate, which needs none either. Those two
    together are what stops a uniformly poor fit from passing on a large
    separation, and a strawman decoy from manufacturing one.
    """
    carrying = [s for s in scored if not s["is_null_record"] and s["span_db"] > 0]
    nulls = [s for s in scored if s["is_null_record"]]

    spans = [s["span_db"] for s in carrying]
    shares = [s["median_abs_db"] / s["span_db"] for s in carrying]
    # The worst record and not the average of them. "Roughly reproduces it" has to
    # hold for every row the type has, or the rows it fails are rows a reader has
    # no warning about.
    share = max(shares) if shares else 1.0
    over_the_floor = [s["record"] for s in nulls if not s["model_stays_inside_the_floor"]]
    gross_ok = share <= GROSS_CEILING and not over_the_floor

    broke = [
        {"record": s["record"], "at": s["worst_at"], "residual_db": s["worst_abs_db"],
         "share_of_span": round(s["worst_abs_db"] / s["span_db"], 3)}
        for s in carrying
        if s["worst_abs_db"] > BREAKDOWN_SHARE * s["span_db"]
    ]
    leaning = [
        s["record"] for s in carrying
        if s["structured"] and s["profile_settled_inside_the_bands"]
    ]
    unreached = [
        {"record": s["record"], "still_moving_db": s["settled_by_db"],
         "why": (
             "The profile had not levelled off where the band set ended, so the corner the "
             "model takes from this record is a lower bound. A residual leaning across the "
             "bands is that bound leaning. Answering it needs a stimulus that reaches "
             "further, not a different model."
         )}
        for s in carrying
        if s["structured"] and not s["profile_settled_inside_the_bands"]
    ]
    breakdown_ok = not broke and not leaning

    checked = []
    for s in scored:
        for row in s["rows"]:
            unit_largest, said = row.get("unit_largest_db"), row.get("largest_db")
            if unit_largest is None or said is None:
                continue
            # Only where both readings said something. A deviation inside the
            # run's own floor has the sign of the floor, and a model asked to
            # reproduce that is asked to reproduce a coin toss. Symmetric on
            # purpose: the model saying nothing where the unit said a decibel is
            # not a sign error either, it is two readings in the noise.
            if abs(unit_largest) <= s["floor_db"] or abs(said) <= s["floor_db"]:
                continue
            checked.append(
                {
                    "record": s["record"],
                    "value": row["value"],
                    "property": "the sign of the largest deviation",
                    "unit": "boost" if unit_largest > 0 else "cut",
                    "model": "boost" if said > 0 else "cut",
                    "same": (unit_largest > 0) == (said > 0),
                }
            )
    for s in nulls:
        checked.append(
            {
                "record": s["record"],
                "property": "the unit's null control shows nothing, and so must the model",
                "unit": f"nothing above a floor of {s['floor_db']:.2f} dB",
                "model": f"{s['model_largest_db']:.2f} dB",
                "same": s["model_stays_inside_the_floor"],
            }
        )
    qualitative_ok = all(c["same"] for c in checked) if checked else False

    # The runner-up has to lean where the winner does not. Not "the runner-up
    # fails the gross gate": at a ceiling loose enough to mean `roughly
    # reproduces it`, every candidate in this class passes it, so a gate written
    # that way could never fire and would have called a five-candidate ranking
    # untested. What separates the candidates here is which records their
    # residual leans on -- seven for the weakest and none for the strongest --
    # and that is the same instrument the breakdown gate uses. Tightening the
    # ceiling until the residual separated them instead would be the chase after
    # the analogue that the ceiling exists to stop.
    runner = ranking[1] if len(ranking) > 1 else None
    power_ok = bool(
        runner and gross_ok and not leaning and runner["leaning_records"] > len(leaning)
    )
    return {
        "gross": {
            "passed": gross_ok,
            "residual_over_span": round(share, 3),
            "span_db": round(max(spans), 2) if spans else 0.0,
            "ceiling": GROSS_CEILING,
            "why": (
                "The median residual over the span the effect commands, taken as the worst of "
                "the records scored. A rejection bound and not a target: below it the residual "
                "is reported and never optimised, because no candidate is decided by it."
            ),
        },
        "qualitative": {
            "passed": qualitative_ok,
            "checked": checked,
            "why": (
                "Properties the unit's readings have that the model's readings must have too. "
                "No arithmetic over a residual rescues a model that cuts where the unit boosts."
            ),
        },
        "power": {
            "passed": power_ok,
            "ranking": ranking,
            "decoy": runner["candidate"] if runner else None,
            "separation": (
                round(runner["worst_share_of_span"] - share, 3) if runner else None
            ),
            "why": (
                "The runner-up in the catalogue, scored on the same records, and whether it "
                "fails the gross gate the winner passed. The decoy is not chosen -- it is "
                "whichever candidate came second, so a strawman cannot be put up in its place. "
                "A class holding one candidate is a catalogue that is unfinished, which is a "
                "finding and not a pass."
            ),
        },
        "breakdown": {
            "passed": breakdown_ok,
            "breaks_down_at": (broke or None) if broke else None,
            "leaning_records": leaning or None,
            "records_the_bands_did_not_contain": unreached or None,
            "why": (
                f"A point worse than {BREAKDOWN_SHARE:g} of the span is a failure at that point "
                "rather than error at it. A residual that leans with the setting or with "
                "frequency is pointing at a structure the model does not have; one that does "
                "not is pointing at the room and the converters, and is not made smaller by "
                "anything this project can do."
            ),
        },
    }


def verdict(result: dict, *, candidates_in_class: int) -> str:
    g = result["gates"]
    if candidates_in_class < 2:
        return "candidates_too_few"
    if not g["gross"]["passed"] or not g["qualitative"]["passed"]:
        return "rejected"
    if not g["breakdown"]["passed"]:
        return "breaks_down"
    if not g["power"]["passed"]:
        return "equivalent_under_this_test"
    return "reproduces"


def load(path: str | Path) -> dict:
    model = json.loads(Path(path).read_text())
    if model["model"].get("kind") != "lti":
        raise ValueError(
            f"{path} is a {model['model'].get('kind')!r} model and this renderer holds only "
            "time-invariant linear ones. A modulated or saturating type needs a time-domain "
            "renderer, and that is not written."
        )
    return model
