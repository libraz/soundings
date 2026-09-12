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

**Two kinds of model, and one set of gates over both.** An `lti` model is a chain
of shelves, peaking sections and gains evaluated as a frequency response, held
against a band profile in decibels. A `table` model is a byte turned into a
quantity, held against what the unit was measured to do at that byte -- for a
rate, in octaves, because a tenth of a hertz is a different error at ten hertz
than at a tenth of one. The gates do not know which they are looking at: every
one of them is a residual against the span the effect commands, and the unit that
span is measured in travels with the record.

**What this renderer cannot do yet.** A modulated or saturating type whose
*waveform* has to be produced needs a time-domain renderer, and the honest state
of that is that it is not written. That is why a rate is identified from the
frequency the archive published rather than from a rendered chorus. A model whose
`kind` is neither `lti` nor `table` is refused rather than approximated.
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


def _structured(rows: list[dict], floor: float, unit: str = "dB") -> tuple[bool, str]:
    """Whether the residual leans, which is the only thing about it that decides.

    Two axes and one rule for both: the residual is structured when the signed
    median over one axis moves across that axis by more than the run's own floor.
    A model that is half a decibel out everywhere is not leaning; a model that is
    half a decibel out at one end of a byte and half the other way at the other
    end is, and the two have the same median.

    **The yardstick is the run's floor and not the residual's own size.** Against
    its own size, a model that fits closely is held to a tighter lean than one
    that fits poorly, which is backwards -- and it makes every good model fail,
    because a good model's residual is small enough that any systematic tenth of
    a decibel exceeds it. The floor is what the same run measured between repeats
    of one setting, so a lean under it is a lean this measurement cannot see.

    Nothing here is about decibels. `unit` names what the residual was measured
    in so the sentence returned says it; a rate is compared in octaves, because a
    tenth of a hertz at ten hertz and at a tenth of a hertz are not the same
    error and a table right at the top of its range and wrong at the bottom would
    otherwise pass.
    """
    if not rows:
        return False, "no readings to say"
    # Medians and not means. A lean is a systematic shift and a mean is moved by
    # the few bands where the take sat near the floor, which is not a shift and is
    # exactly where the readings are noisiest. Taking the mean reported a lean of
    # one and a half decibels on the level row whose own map had been transcribed
    # from that record: an artefact of the statistic, not of the model.
    per_setting = [float(np.median(row["residual"])) for row in rows]
    setting_swing = float(max(per_setting) - min(per_setting))
    widths = {len(row["residual"]) for row in rows}
    if len(widths) == 1:
        stacked = np.array([row["residual"] for row in rows], dtype=float)
        per_band = np.median(stacked, axis=0)
        band_swing = float(np.nanmax(per_band) - np.nanmin(per_band))
    else:
        band_swing = 0.0
    leans = setting_swing > floor or band_swing > floor
    return leans, (
        f"the signed median swings {setting_swing:.4g} {unit} across the settings and "
        f"{band_swing:.4g} {unit} across the readings within one, against the run's own "
        f"floor of {floor:.4g} {unit}"
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

    floors = record.get("reference", {}).get("floor_db") or [0.0]
    floor = float(max(floors))

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
        answered = reading["band_db"]
        residual = [round(said[i] - answered[i], 2) for i in usable]
        rows.append(
            {
                "value": value,
                "model_reading": [said[i] for i in usable],
                "unit_reading": [answered[i] for i in usable],
                "residual": residual,
                "model_largest": max((said[i] for i in usable), key=abs, default=None),
                "unit_largest": reading.get("largest_db"),
            }
        )

    flat = np.array([abs(v) for row in rows for v in row["residual"]], dtype=float)
    every = [v for row in rows for v in row["unit_reading"]]
    span = float(max(every) - min(every)) if every else 0.0
    median_abs = float(np.median(flat)) if flat.size else 0.0
    worst = float(flat.max()) if flat.size else 0.0
    worst_where = None
    if flat.size:
        for row in rows:
            for j, v in enumerate(row["residual"]):
                if abs(v) == worst:
                    worst_where = {"value": row["value"], "hz": [centres[i] for i in usable][j]}
                    break
            if worst_where:
                break
    leans, why_leans = _structured(rows, floor)
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
    is_null = span <= 2.0 * floor
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
        (abs(v) for row in rows for v in row["model_reading"]), default=0.0
    )
    return {
        "record": None,
        "address": record.get("address"),
        "measured_in": "dB",
        "span": round(span, 2),
        "floor": round(floor, 2),
        "is_null_record": is_null,
        "settled_by": round(unsettled, 2),
        "reading_is_a_value_not_a_bound": bool(unsettled <= floor),
        "model_largest": round(model_largest, 2),
        "model_stays_inside_the_floor": bool(model_largest <= floor),
        "median_abs": round(median_abs, 3),
        "worst_abs": round(worst, 3),
        "worst_at": worst_where,
        "structured": leans,
        "why_structured": why_leans,
        "bands_above_the_models_nyquist_hz": dropped,
        "readings_left_out": left_out,
        "rows": rows,
    }


# ---------------------------------------------------------------- a byte as a table


def rate_of(model: dict, printed_range: str, byte_value: int) -> float:
    """What one candidate says the modulator runs at, at one setting of the byte.

    The printed range picks the table because that is the only thing outside the
    unit that distinguishes one rate slot from another, and it is what a part
    built in this era would have been given: two tables in ROM and a pointer per
    effect. It is also read off a page rather than fitted, so every candidate gets
    it and none is helped by it.
    """
    table = model["tables"][printed_range]
    kind = table["kind"]
    first, top = float(table["first_hz"]), float(table.get("top_hz", 0.0))
    if kind == "steps":
        # Walked rather than solved, because the claim is that these are entries
        # and not a formula: a run that ends and hands over to the next is what a
        # printed list of steps looks like from the inside.
        here = first
        for v in range(1, byte_value + 1):
            step = next(
                (run["step_hz"] for run in table["runs"] if v <= run["through"]),
                None,
            )
            if step is None:
                return here  # past the last entry, and the table holds its last
            here += step
        return here
    if kind == "linear":
        return first + (top - first) * byte_value / 127.0
    if kind == "geometric":
        return first * (top / first) ** (byte_value / 127.0)
    raise ValueError(f"{kind!r} is not a table this renderer knows")


def _rate_floor(readings: list[dict]) -> float:
    """What the run itself resolved a rate to, in octaves.

    Each reading carries the figure every partial of the take found. Their spread
    is the run's own disagreement with itself at one setting, which is the only
    floor available here and the one the lean has to clear. The median of those
    spreads and not the largest: one take that lost a partial would otherwise set
    the floor for the whole sweep and hide a real lean behind it.
    """
    spreads = []
    for reading in readings:
        rates = [r for r in (reading.get("rates") or []) if r and r > 0]
        if len(rates) < 2:
            continue
        spreads.append(np.log2(max(rates) / min(rates)))
    return float(np.median(spreads)) if spreads else 0.0


def admitted_rates(record: dict) -> tuple[list[dict], list[dict]]:
    """The readings of a rate record that the record's own controls stand behind.

    Two exclusions and the record states both of them itself. A figure fewer than
    half the partials found is not "the one the partials agreed on", which is what
    the method says it reports. A figure at or under the take's own slowest
    measurable rate is the floor of the analysis and not a rate, which is what the
    limits say -- and it is stable, which is what makes it dangerous.

    Neither looks at any model. Both are named per reading so a reader can see how
    much of a sweep was dropped and why.
    """
    kept, left_out = [], []
    for reading in record["readings"]:
        hz = reading.get("rate_hz")
        if hz is None or hz <= 0:
            continue
        agreeing, of = reading.get("agreeing", 0), reading.get("of", 1)
        slowest = reading.get("slowest_measurable_hz")
        if agreeing * 2 <= of:
            left_out.append(
                {
                    "value": reading["value"],
                    "why": (
                        f"{agreeing} of {of} partials found it, so it is not what "
                        "they agreed on"
                    ),
                }
            )
            continue
        if slowest is not None and hz <= slowest:
            left_out.append(
                {
                    "value": reading["value"],
                    "why": (
                        f"the take was long enough for {slowest:g} Hz and the reading is "
                        f"{hz:g} Hz, so what it reports is that limit"
                    ),
                }
            )
            continue
        kept.append(reading)
    return kept, left_out


def heard_one_modulator(
    record: dict, *, rate_slots_printed_for_this_type: int
) -> tuple[bool, str]:
    """Whether a rate record is evidence about the byte it swept.

    A take is one recording of everything the effect is doing. The archive's own
    note says a type with more than one modulator returns whichever dominates and
    that nothing in the numbers says which is which, so a slot on such a type is
    not evidence about its own byte until the other stage is silenced -- and the
    remedy is a run, which is what makes this a queue entry rather than a verdict.

    The second test is that the readings rise with the byte. They need not match
    anything; they need only move the way any rate table moves, which every
    candidate in the catalogue agrees on. A slot whose readings fall somewhere is
    a take that changed what it was listening to part way through the sweep.

    Both tests are blind to the model, and neither can be satisfied by a fit.
    """
    if rate_slots_printed_for_this_type != 1:
        return False, (
            f"the page prints {rate_slots_printed_for_this_type} rates for this type, and a take "
            "returns whichever modulator dominates it; which one these readings are of is not in "
            "the numbers"
        )
    kept, left_out = admitted_rates(record)
    order = sorted({(int(r["value"]), float(r["rate_hz"])) for r in kept})
    if len({v for v, _ in order}) < 2:
        # One reading cannot show a step, and none cannot show anything. Counting
        # such a slot as evidence would put a number in the tally that no gate can
        # act on, which reads as corroboration and is an absence.
        return False, (
            f"the record's own controls stand behind {len(order)} of its "
            f"{len(order) + len(left_out)} readings, and a step needs two settings"
        )
    floor = _rate_floor(kept)
    for (v1, hz1), (v2, hz2) in zip(order, order[1:], strict=False):
        if np.log2(hz1 / hz2) > floor:
            return False, (
                f"the reading falls from {hz1:g} Hz at {v1} to {hz2:g} Hz at {v2}, past the run's "
                f"own floor of {floor:.4g} octaves, so the take stopped following one thing"
            )
    return True, "one printed rate, and the readings rise with the byte"


def score_against_rates(model: dict, record: dict, *, printed_range: str) -> dict:
    """One published rate curve, answered by a table.

    In octaves throughout. A tenth of a hertz is most of the effect at the bottom
    of this range and nothing at the top, so a comparison in hertz would pass a
    table that is right where the numbers are large and wrong where they are
    small -- which is the half a player hears as the slow setting.
    """
    kept, left_out = admitted_rates(record)
    run_floor = _rate_floor(kept)
    # What one entry of this table is worth, at the settings this slot was read
    # at. The lean is judged against the coarser of the two floors, because a
    # residual smaller than one entry is a residual no candidate in this class can
    # differ over: entries are what they are all made of, and moving one by less
    # than a step is not a model anybody could write.
    #
    # This is only safe because the step is measured rather than assumed. Every
    # one of the 128 settings was read at one slot, so there is no room between
    # them for entries a coarser sweep would have missed. Where that is not true
    # of a class, this floor would hide a finer table and must not be used.
    steps = []
    for reading in kept:
        value = int(reading["value"])
        if value == 0:
            continue
        before = rate_of(model, printed_range, value - 1)
        after = rate_of(model, printed_range, value)
        if after > before > 0:
            steps.append(float(np.log2(after / before)))
    model_floor = float(np.median(steps)) if steps else 0.0
    floor = max(run_floor, model_floor)
    rows = []
    slowest_value = min((int(r["value"]) for r in kept), default=0)
    base_unit = next(
        (float(r["rate_hz"]) for r in kept if int(r["value"]) == slowest_value), 1.0
    )
    base_model = rate_of(model, printed_range, slowest_value)
    for reading in sorted(kept, key=lambda r: int(r["value"])):
        value = int(reading["value"])
        said = rate_of(model, printed_range, value)
        answered = float(reading["rate_hz"])
        rows.append(
            {
                "value": value,
                "model_reading": [round(float(np.log2(said)), 5)],
                "unit_reading": [round(float(np.log2(answered)), 5)],
                "residual": [round(float(np.log2(said / answered)), 5)],
                "model_largest": round(float(np.log2(said / base_model)), 5),
                "unit_largest": round(float(np.log2(answered / base_unit)), 5),
                "model_hz": round(said, 4),
                "unit_hz": round(answered, 4),
            }
        )

    flat = np.array([abs(row["residual"][0]) for row in rows], dtype=float)
    every = [row["unit_reading"][0] for row in rows]
    span = float(max(every) - min(every)) if every else 0.0
    worst = float(flat.max()) if flat.size else 0.0
    worst_where = next(
        ({"value": row["value"], "hz": row["unit_hz"]} for row in rows
         if abs(row["residual"][0]) == worst),
        None,
    )
    leans, why_leans = _structured(rows, floor, "octaves")
    model_largest = max((abs(row["model_largest"]) for row in rows), default=0.0)
    return {
        "record": None,
        "address": record.get("address"),
        "measured_in": "octaves",
        "printed_range": printed_range,
        "span": round(span, 4),
        "floor": round(floor, 5),
        "floor_the_run_resolved": round(run_floor, 6),
        "floor_of_one_entry": round(model_floor, 6),
        "is_null_record": span <= 2.0 * floor,
        "settled_by": 0.0,
        # No escape hatch on this path. On the band records one exists because the
        # record publishes a figure saying the profile had not levelled off inside
        # the band set, so the corner taken from it is a bound. A rate record
        # publishes no such figure, and inventing one here would turn every lean
        # into a measurement's limit, which is the loophole the gates exist to
        # close.
        "reading_is_a_value_not_a_bound": True,
        "sign_property": "faster or slower than the slowest setting of the byte",
        "above": "faster",
        "below": "slower",
        "model_largest": round(model_largest, 4),
        "model_stays_inside_the_floor": bool(model_largest <= floor),
        "median_abs": round(float(np.median(flat)), 5) if flat.size else 0.0,
        "worst_abs": round(worst, 5),
        "worst_at": worst_where,
        "structured": leans,
        "why_structured": why_leans,
        "readings_left_out": left_out,
        "rows": rows,
    }


# ---------------------------------------------------------------- the gates


def gates(scored: list[dict], *, ranking: list[dict]) -> dict:
    """The four gates, over every record the model was scored against.

    No gate on the residual's absolute size. The gross gate is the residual
    against the span the effect itself commands, which is measured per record and
    needs no figure to be invented; the power gate is whether the runner-up in the
    catalogue fails the same gross gate, which needs none either. Those two
    together are what stops a uniformly poor fit from passing on a large
    separation, and a strawman decoy from manufacturing one.
    """
    units = {s.get("measured_in", "dB") for s in scored}
    if len(units) > 1:
        raise ValueError(
            f"these records were scored in {sorted(units)} and one gate cannot span them: a "
            "share of a span means nothing across two units, and the lean would be compared "
            "against the wrong floor"
        )
    unit_name = units.pop() if units else "dB"
    carrying = [s for s in scored if not s["is_null_record"] and s["span"] > 0]
    nulls = [s for s in scored if s["is_null_record"]]

    spans = [s["span"] for s in carrying]
    shares = [s["median_abs"] / s["span"] for s in carrying]
    # The worst record and not the average of them. "Roughly reproduces it" has to
    # hold for every row the type has, or the rows it fails are rows a reader has
    # no warning about.
    share = max(shares) if shares else 1.0
    over_the_floor = [s["record"] for s in nulls if not s["model_stays_inside_the_floor"]]
    gross_ok = share <= GROSS_CEILING and not over_the_floor

    broke = [
        {"record": s["record"], "at": s["worst_at"], "residual": s["worst_abs"],
         "share_of_span": round(s["worst_abs"] / s["span"], 3)}
        for s in carrying
        if s["worst_abs"] > BREAKDOWN_SHARE * s["span"]
    ]
    leaning = [
        s["record"] for s in carrying
        if s["structured"] and s["reading_is_a_value_not_a_bound"]
    ]
    unreached = [
        {"record": s["record"], "still_moving": s["settled_by"],
         "why": (
             "The profile had not levelled off where the band set ended, so the corner the "
             "model takes from this record is a lower bound. A residual leaning across the "
             "bands is that bound leaning. Answering it needs a stimulus that reaches "
             "further, not a different model."
         )}
        for s in carrying
        if s["structured"] and not s["reading_is_a_value_not_a_bound"]
    ]
    breakdown_ok = not broke and not leaning

    checked = []
    for s in scored:
        for row in s["rows"]:
            unit_largest, said = row.get("unit_largest"), row.get("model_largest")
            if unit_largest is None or said is None:
                continue
            # Only where both readings said something. A deviation inside the
            # run's own floor has the sign of the floor, and a model asked to
            # reproduce that is asked to reproduce a coin toss. Symmetric on
            # purpose: the model saying nothing where the unit said a decibel is
            # not a sign error either, it is two readings in the noise.
            if abs(unit_largest) <= s["floor"] or abs(said) <= s["floor"]:
                continue
            checked.append(
                {
                    "record": s["record"],
                    "value": row["value"],
                    "property": s.get("sign_property", "the sign of the largest deviation"),
                    "unit": s.get("above", "boost") if unit_largest > 0 else s.get("below", "cut"),
                    "model": s.get("above", "boost") if said > 0 else s.get("below", "cut"),
                    "same": (unit_largest > 0) == (said > 0),
                }
            )
    for s in nulls:
        checked.append(
            {
                "record": s["record"],
                "property": "the unit's null control shows nothing, and so must the model",
                "unit": f"nothing above a floor of {s['floor']:.4g} {unit_name}",
                "model": f"{s['model_largest']:.4g} {unit_name}",
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
        "measured_in": unit_name,
        "gross": {
            "passed": gross_ok,
            "residual_over_span": round(share, 3),
            "span": round(max(spans), 3) if spans else 0.0,
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
            "records_the_run_did_not_reach": unreached or None,
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


RENDERED = ("lti", "table")
"""The kinds of model this module can hold against the archive.

Named rather than open so that a class nobody has written a renderer for cannot
be scored by accident. A saturating type would need its harmonics produced and a
modulated one its waveform, and neither is written; a model claiming to be either
is refused at the door instead of being fitted with the wrong instrument.
"""


def load(path: str | Path) -> dict:
    model = json.loads(Path(path).read_text())
    kind = model["model"].get("kind")
    if kind not in RENDERED:
        raise ValueError(
            f"{path} is a {kind!r} model and this renderer holds only time-invariant linear "
            "ones and tables. A modulated or saturating type needs a time-domain renderer, "
            "and that is not written."
        )
    return model
