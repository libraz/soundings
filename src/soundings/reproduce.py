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

**What this renderer cannot do yet.** A model that claims a structure whose
*waveform* has to be produced -- the network a reverb is, the curve a saturating
stage bends by -- needs a time-domain renderer, and the honest state of that is
that it is not written. A model whose `kind` is neither `lti` nor `table` is
refused rather than approximated.

**That is a narrower gap than it sounds, and reading it as a wide one held work
up.** What a byte selects is a quantity, and a quantity is answered by a table
against a curve the archive published -- whatever the stage holding it does to a
waveform. A rate was identified on twenty-one types without a chorus being
rendered, and a delay time is a time whether or not the delay modulates. So a
class with no reading is waiting on a stage that publishes its quantity, which is
a measurement; only a class whose claim is the structure itself is waiting on this.
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

    Five rules, and each of them is a shape this archive has measured rather than
    a shape a model wanted. A byte that answers over a window and sticks at both
    ends, a byte that has two states, a byte read at some values and interpolated
    between them, and a byte that indexes a short table and returns the first
    entry for anything past its end -- all four are in the equaliser alone. The
    fifth is a byte whose top bits index a table and whose bottom bits do nothing,
    which is a divide the era did not have to write: a stride that is a power of
    two is a shift, and a table read that way holds its last entry rather than
    running off the end.

    `points` with `log` set to true and only the two printed ends in it is how a
    candidate says the byte is continuous, so nothing here needs a rule for that.
    """
    kind = spec["kind"]
    if kind == "stepped-table":
        entries = spec["entries"]
        stride = int(spec["per_entry"])
        index = min(byte_value // stride, len(entries) - 1)
        return float(entries[index])
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


def _peak_band(profile: list[float], usable: list[int], centres: list[float]) -> float | None:
    """The band a deviation profile is largest in, by absolute size.

    Absolute because a cut is as much a feature as a boost, and a byte that moves
    a notch is the same question as one that moves a peak.
    """
    if not usable:
        return None
    values = [abs(profile[i]) for i in usable]
    return float(centres[int(np.argmax(values))])


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
        here = [centres[i] for i in usable]
        rows.append(
            {
                "value": value,
                "model_reading": [said[i] for i in usable],
                "unit_reading": [answered[i] for i in usable],
                "residual": residual,
                "model_largest": max((said[i] for i in usable), key=abs, default=None),
                "unit_largest": reading.get("largest_db"),
                # Which band each profile is largest in, both read off the same set.
                # The band and not a frequency: a peak is located to a band here, and
                # the model's figure goes through the same takes and the same window,
                # so whatever the reading does to a broad peak it does to both.
                "model_peak_hz": _peak_band(said, usable, here),
                "unit_peak_hz": _peak_band(answered, usable, here),
                # And where a fit over the whole feature puts each of them, which is
                # the same question asked without the band quantising the answer.
                # The archive's own reading, applied to both sides here rather than
                # taken from the record on one side and computed on the other.
                "model_fitted_hz": efxbands.fitted_position(
                    [said[i] for i in usable], here
                ),
                "unit_fitted_hz": efxbands.fitted_position(
                    [answered[i] for i in usable], here
                ),
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
        "band_width_octaves": width,
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
        # Band by band, so that a reading taken at one band can be given the floor
        # of that band. `floor` above is the worst of these and is right for a
        # residual spread over the whole profile; it is not right for a figure read
        # at the top of a feature, where a band the feature never reaches would
        # otherwise set what the reading is taken to resolve.
        "floor_by_band_db": [round(float(f), 3) for f in floors],
        "bands_hz": centres,
        "readings_left_out": left_out,
        "rows": rows,
    }


# ---------------------------------------------------------------- a byte as a place


LOCATED_BY = {"band": "peak_hz", "fit": "fitted_hz"}
"""The two readings of where a feature sits, and the field each of them lands in.

`band` is which band the profile is largest in. It is quantised to a band and
cannot be finer however good the takes are, and it carries no bias: whatever the
band set does to a broad feature it does to whichever profile went through it.

`fit` is the top of a curve fitted over the feature's whole width. It lands
between bands and one band's scatter moves it by a fraction of what it moves the
largest -- and it is pulled towards the longer flank of a feature that is not
symmetric, which a peaking filter is not once the bilinear transform has squeezed
its upper side. Both sides of a comparison go through whichever is asked for, so
either bias cancels; what does not cancel is reading one side one way.
"""


def score_against_peaks(scored: dict, *, located_by: str = "band") -> dict:
    """The same rendering read again, as where the feature sits rather than how big it is.

    A residual in decibels is the wrong instrument for a byte that moves a feature
    along the frequency axis. Sliding a peak of moderate width by a sixth of an
    octave costs a decibel or two on its flanks and nothing at its top, which is
    under the run's own floor over most of the profile -- so a byte read this way
    reports every candidate as fitting and none as leaning, which is not that the
    candidates agree but that the reading cannot hear them. Four candidates
    separated by two bands at every setting came back `equivalent_under_this_test`
    that way.

    Both peaks are read off the same band set, from the same takes, through the
    same window: the model's profile is rendered and then measured exactly as the
    unit's was. Whatever a band-energy reading does to a broad peak it does to
    both of them, so the comparison carries no correction and needs none.

    The floor is one band, which is what the reading resolves and what the record
    publishes. There is no second floor here. One entry of the candidate's table
    would be a coarser one and would swallow the disagreement, but that floor is
    only admissible where the step has been measured -- and on this class the
    stride is the question, so assuming it would be assuming the answer.

    A fitted position is finer than that and keeps the same floor, which is then a
    bound and not a resolution: how far apart the same setting's own takes put a
    fitted position is not measured anywhere in this archive, so the number that is
    measured is used instead. It errs towards calling a disagreement noise, which
    is the direction to err in.
    """
    if located_by not in LOCATED_BY:
        raise ValueError(f"a feature is located by {' or '.join(LOCATED_BY)}, not {located_by!r}")
    field = LOCATED_BY[located_by]
    rows = [
        r for r in scored["rows"]
        if r.get(f"unit_{field}") and r.get(f"model_{field}")
    ]
    floor = float(scored["band_width_octaves"])
    out = []
    for row in rows:
        unit, said = float(row[f"unit_{field}"]), float(row[f"model_{field}"])
        out.append(
            {
                "value": row["value"],
                "unit_reading": [unit],
                "model_reading": [said],
                "residual": [float(np.log2(said / unit))],
                "bands_apart": round(float(np.log2(said / unit)) / floor, 2),
            }
        )
    flat = np.array([abs(r["residual"][0]) for r in out], dtype=float)
    heard = [r["unit_reading"][0] for r in out]
    span = float(np.log2(max(heard) / min(heard))) if len(heard) > 1 else 0.0
    leans, why = _structured(out, floor, unit="octaves")
    return {
        "record": scored.get("record"),
        "address": scored.get("address"),
        "measured_in": "octaves",
        "located_by": located_by,
        "span": round(span, 4),
        "floor": round(floor, 4),
        "is_null_record": bool(scored["is_null_record"]),
        "settled_by": 0.0,
        "reading_is_a_value_not_a_bound": True,
        "model_largest": 0.0,
        "model_stays_inside_the_floor": True,
        "median_abs": round(float(np.median(flat)), 4) if flat.size else 0.0,
        "worst_abs": round(float(flat.max()), 4) if flat.size else 0.0,
        "worst_at": (
            {"value": max(out, key=lambda r: abs(r["residual"][0]))["value"]} if out else None
        ),
        "structured": leans,
        "why_structured": why,
        "readings_left_out": [],
        "properties": _which_settings_moved(out, floor),
        "rows": out,
    }


# ------------------------------------------------------------- a byte as a height


def _floor_where_the_feature_is(scored: dict, rows: list[dict]) -> float:
    """The run's own spread, at the bands the unit's feature was largest in.

    The worst of them, over the settings scored, so one setting whose peak happens
    to land in a quiet band does not set the floor for the rest. Where a record
    does not carry its floor band by band the profile's own worst is used, which is
    the older behaviour and errs towards calling a disagreement noise.
    """
    per_band = scored.get("floor_by_band_db")
    centres = scored.get("bands_hz")
    if not per_band or not centres:
        return float(scored["floor"])
    seen = []
    for row in rows:
        at = row.get("unit_peak_hz")
        if at is None:
            continue
        nearest = min(range(len(centres)), key=lambda i: abs(np.log2(centres[i] / at)))
        seen.append(float(per_band[nearest]))
    return max(seen) if seen else float(scored["floor"])


def score_against_heights(scored: dict) -> dict:
    """The same rendering read again, as how big the feature is rather than as a profile.

    A residual taken over every band is the wrong instrument for a byte whose whole
    quantity is one height. Most of a band set is skirt: a gain byte that is out by
    half a decibel at the top of its feature is out by a tenth of that two octaves
    away, and there are twenty such bands against the two or three that carry the
    height. The median over the profile is then mostly a reading of the bands where
    every candidate agrees, and a candidate wrong by two decibels where it matters
    ranks ahead of one that is right there -- which is what the gain class returned
    before this was written, with the winner leaning and the runner-up not.

    Both heights are read off the same band set, from the same takes, through the
    same window, so whatever a band-energy reading takes off the top of a broad
    feature it takes off both. A section whose peak the band set reads at ninety-nine
    hundredths of its true height gives back ninety-nine hundredths on either side of
    the comparison, and the ratio is the whole of what is compared here.

    The floor is the record's own spread of its repeats, taken at the bands the
    feature was actually largest in rather than across the whole profile. A height
    read at one kilohertz is not bounded by how far the repeats scattered at twelve
    and a half, where the feature is nothing and the takes are nearest the floor --
    quoting the worst band anywhere as what this reading resolves is a limit of the
    reading being written down as a property of the unit, and it is over a decibel
    on records whose peak band repeats to three hundredths.

    There is no second floor. One entry of a candidate's table would be coarser and
    would swallow the disagreement, and on this class the entries are the question.
    """
    # A height needs a feature. At the setting a run took its reference from, the
    # deviation is nothing and the largest band is whichever one the noise won --
    # a number that is a reading of the room, against a model that correctly says
    # zero. Three records leaned on that single row and on nothing else. Left out
    # by name rather than dropped: what the model owes such a setting is that it
    # shows nothing there too, which `model_stays_inside_the_floor` already says.
    coarse = float(scored["floor"])
    rows, left_out = [], list(scored.get("readings_left_out", []))
    for row in scored["rows"]:
        if row.get("unit_largest") is None or row.get("model_largest") is None:
            continue
        if abs(float(row["unit_largest"])) <= coarse:
            left_out.append(
                {
                    "value": row["value"],
                    "why": (
                        f"the unit's profile reaches {abs(float(row['unit_largest'])):.2f} dB "
                        f"here against the run's own floor of {coarse:.2f}, so there is no "
                        "feature to read a height off"
                    ),
                }
            )
            continue
        rows.append(row)
    floor = _floor_where_the_feature_is(scored, rows)
    out = []
    for row in rows:
        unit, said = float(row["unit_largest"]), float(row["model_largest"])
        out.append(
            {
                "value": row["value"],
                "unit_reading": [unit],
                "model_reading": [said],
                "residual": [said - unit],
                "floors_apart": round((said - unit) / floor, 2) if floor else None,
            }
        )
    flat = np.array([abs(r["residual"][0]) for r in out], dtype=float)
    heard = [r["unit_reading"][0] for r in out]
    span = float(max(heard) - min(heard)) if len(heard) > 1 else 0.0
    leans, why = _structured(out, floor)
    return {
        "record": scored.get("record"),
        "address": scored.get("address"),
        "measured_in": "dB",
        "read_as": "height",
        "span": round(span, 3),
        "floor": round(floor, 3),
        "is_null_record": bool(scored["is_null_record"]),
        "settled_by": scored.get("settled_by", 0.0),
        "reading_is_a_value_not_a_bound": bool(scored.get("reading_is_a_value_not_a_bound", True)),
        "model_largest": scored.get("model_largest", 0.0),
        "model_stays_inside_the_floor": bool(scored.get("model_stays_inside_the_floor", False)),
        "median_abs": round(float(np.median(flat)), 4) if flat.size else 0.0,
        "worst_abs": round(float(flat.max()), 4) if flat.size else 0.0,
        "worst_at": (
            {"value": max(out, key=lambda r: abs(r["residual"][0]))["value"]} if out else None
        ),
        "structured": leans,
        "why_structured": why,
        "readings_left_out": left_out,
        "properties": _which_settings_moved(out, floor, as_ratio=False),
        "rows": out,
    }


def _which_settings_moved(
    rows: list[dict], floor: float, *, as_ratio: bool = True
) -> list[dict]:
    """Which neighbouring settings the reading tells apart at all, asked of both.

    This is the property that separates a table from a curve and one stride from
    another, and it survives an error in where the table sits: a candidate whose
    entries are all a band low still repeats in the same places. It is stated as
    a property rather than folded into the residual because those are two
    different disagreements -- one says the table is in the wrong place, the other
    says it does not have this many entries -- and a model can be wrong in either
    without being wrong in the other.

    Against the floor and not by equality. Two settings of one entry are the same
    number to the model, which renders them from the same takes and cannot differ,
    and two takes to the unit, which can. Read at a twelfth of an octave the two
    land in one band and equality would hold; read four times finer they do not,
    because the top of a peak of this width is flat enough that a tenth of a
    decibel moves which band is largest. Asking whether the reading tells them
    apart rather than whether it repeats itself is the same question at every
    resolution, and it is the question the rest of this file asks.
    """
    # Where the quantity is a place, two settings are apart by the ratio between
    # them and the floor is a width in octaves. Where it is a height, they are
    # apart by the difference and the floor is a spread in decibels -- and a ratio
    # would be meaningless there, because a height passes through zero and changes
    # sign, which a frequency does not.
    def apart(before: dict, after: dict, key: str) -> bool:
        if as_ratio:
            return bool(abs(np.log2(after[key][0] / before[key][0])) > floor)
        return bool(abs(after[key][0] - before[key][0]) > floor)

    out = []
    for before, after in zip(rows, rows[1:], strict=False):
        unit_apart = apart(before, after, "unit_reading")
        model_apart = apart(before, after, "model_reading")
        out.append(
            {
                "value": after["value"],
                "property": (
                    f"the reading tells settings {before['value']} and "
                    f"{after['value']} apart, or it does not"
                ),
                "unit": "told apart" if unit_apart else "not told apart",
                "model": "told apart" if model_apart else "not told apart",
                "same": unit_apart == model_apart,
            }
        )
    return out


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


# ---- a byte as a multiplier

WHY_OWN_FLOOR = (
    "How many settings are further out than that setting alone could be, and by how "
    "much. The floors above are each one number over the whole byte, and this byte "
    "covers three orders of magnitude: half a quefrency is a fifth of an octave at "
    "its shortest setting and a ten-thousandth of one at its longest. A single floor "
    "is therefore far over the reading at one end and far under it at the other, so a "
    "candidate wrong where the reading is coarse passes it and a candidate right "
    "everywhere can fail it. This counts each setting against its own. It is reported "
    "beside the swing and does not replace it: the swing asks whether the residual "
    "leans, which is a different question from whether any one reading disagrees."
)


def time_of(model: dict, printed_range: str, byte_value: int) -> float:
    """What one candidate says the delay is, in milliseconds, at one setting.

    The printed range picks the table for the same reason it picks a rate's: it is
    the only thing outside the unit that tells one time slot from another, it is
    read off a page rather than fitted, and every candidate gets it. The frequency
    class has since measured that the printed range does pick the table, on three
    ranges and three types, which is an argument from a neighbouring class and not
    a measurement of this one -- so it carries the arrangement here and does not
    close anything about it.

    Expressed in the same vocabulary the other table classes use, so a candidate
    here needs no rule this renderer did not already hold.
    """
    return _from_map(model["tables"][printed_range], int(byte_value))


def admitted_times(record: dict) -> tuple[list[dict], list[dict]]:
    """The readings of a time record the record's own controls stand behind.

    One exclusion and the record states it itself. A cepstrum always has a
    strongest peak, so a setting with no copy in its output still returns a time --
    a short one, a plausible one, and a repeatable one, because the roughness of a
    stimulus is a property of the stimulus. What separates the two is how far the
    peak stood over that roughness, and the record carries the verdict rather than
    this function recomputing it: the bar was set from injected combs, and a bar
    applied here would be one this side of the comparison chose.

    A setting the chain itself puts a peak near is not excluded. It is a reading of
    the effect wherever the effect is louder there, and the record names where the
    chain's own peak sits so a row landing on it can be seen. Dropping those would
    be this side deciding which of the unit's answers count.
    """
    kept, left_out = [], []
    for reading in record["readings"]:
        if reading.get("admitted"):
            kept.append(reading)
            continue
        left_out.append(
            {
                "value": reading["value"],
                "why": (
                    f"the strongest peak stood {reading['stands']:g} times the "
                    f"carrier's own roughness, under the {record['stands_out']:g} an "
                    "injected comb was measured to need, so what it returns is the "
                    "roughness"
                ),
            }
        )
    return kept, left_out


def _time_floors(record: dict, kept: list[dict]) -> tuple[float, float]:
    """The two floors a time record carries, both in octaves.

    The first is what the run resolved: a setting taken more than once, and how far
    apart the reading put it. The second is the grid -- one quefrency of the
    transform the reading was made through, which is the finest difference it could
    have shown whatever the takes were like.

    They are different claims and both are needed. The repeats can come back
    identical, because the answer is quantised to that grid and two takes can land
    in one cell; reporting that as the reading being exact is a limit of the
    measurement published as a property of the unit. The grid cannot stand in for
    the repeats either: it says what the transform could resolve and nothing about
    whether the unit, the room and the converters put the same answer back twice.

    Each is in octaves because that is what the residual is in, and one quefrency
    is most of a short delay and nothing of a long one.
    """
    seen: dict[int, list[float]] = {}
    for reading in kept:
        seen.setdefault(int(reading["value"]), []).append(float(reading["ms"]))
    spreads = [
        float(np.log2(max(v) / min(v))) for v in seen.values() if len(v) > 1 and min(v) > 0
    ]
    run_floor = float(np.median(spreads)) if spreads else 0.0

    step = float(record.get("quefrency_step_ms") or 0.0)
    grid = [
        float(np.log2((r["ms"] + step) / r["ms"]))
        for r in kept
        if float(r["ms"]) > 0
    ]
    return run_floor, (float(np.median(grid)) if grid else 0.0)


def score_against_times(model: dict, record: dict, *, printed_range: str) -> dict:
    """One published delay curve, answered by a table.

    In octaves throughout, for the reason a rate is. This byte covers three orders
    of magnitude -- the printed range begins at nothing and ends at half a second --
    so a comparison in milliseconds would pass a table that is right where the
    numbers are large and wrong where they are small, which is the half a player
    hears as the short setting.
    """
    kept, left_out = admitted_times(record)
    run_floor, grid_floor = _time_floors(record, kept)
    # What one entry of this table is worth, at the settings this slot was read at.
    # The lean is judged against the coarsest of the floors, because a residual
    # smaller than one entry is a residual no candidate in this class can differ
    # over: entries are what they are all made of.
    #
    # This is only safe because the step is measured rather than assumed. The slot
    # this class was read on was asked at every one of its 128 settings, so there is
    # no room between them for entries a coarser sweep would have missed. Where that
    # is not true of a record, this floor would hide a finer table and must not be
    # used.
    steps = []
    for reading in kept:
        value = int(reading["value"])
        if value == 0:
            continue
        before = time_of(model, printed_range, value - 1)
        after = time_of(model, printed_range, value)
        if after > before > 0:
            steps.append(float(np.log2(after / before)))
    model_floor = float(np.median(steps)) if steps else 0.0
    floor = max(run_floor, grid_floor, model_floor)
    step_ms = float(record.get("quefrency_step_ms") or 0.0)

    def floor_here(value: int, answered: float) -> float:
        """What this one setting could differ by, rather than what the run could.

        The three floors above are one number each, and one number cannot be the
        floor of a byte covering three orders of magnitude. Half a quefrency is a
        fifth of an octave at the shortest setting this slot has and a ten-thousandth
        of one at the longest, so a median of the two is far over the floor at one
        end and far under it at the other -- and a lean judged by it is found where
        the reading is coarse and missed where the candidates actually differ.
        """
        grid = 0.0
        if answered > 0 and step_ms > 0:
            below = answered - step_ms / 2.0
            grid = float(np.log2(answered / below)) if below > 0 else float("inf")
        before = time_of(model, printed_range, value - 1) if value else 0.0
        said = time_of(model, printed_range, value)
        entry = float(np.log2(said / before)) if said > before > 0 else 0.0
        return max(run_floor, grid, entry)

    rows = []
    # Settings the comparison cannot be made at, named rather than passed over. A
    # residual is a ratio and neither side of it can be nothing, so a candidate that
    # says a setting is no delay at all has no octave to be wrong by there -- which
    # is a thing the candidate said, not a row that was not read.
    not_comparable = []
    shortest_value = min((int(r["value"]) for r in kept), default=0)
    base_unit = next(
        (float(r["ms"]) for r in kept if int(r["value"]) == shortest_value), 1.0
    )
    base_model = time_of(model, printed_range, shortest_value)
    for reading in sorted(kept, key=lambda r: (int(r["value"]), r.get("take", ""))):
        value = int(reading["value"])
        said = time_of(model, printed_range, value)
        answered = float(reading["ms"])
        if said <= 0 or answered <= 0:
            not_comparable.append(
                {
                    "value": value,
                    "unit_ms": round(answered, 4),
                    "model_ms": round(said, 4),
                    "why": (
                        "a residual here is a ratio of two times, and this candidate "
                        "puts no time on one side of it"
                        if said <= 0
                        else "the record put no time on this setting"
                    ),
                }
            )
            continue
        rows.append(
            {
                "value": value,
                "model_reading": [round(float(np.log2(said)), 5)],
                "unit_reading": [round(float(np.log2(answered)), 5)],
                "residual": [round(float(np.log2(said / answered)), 5)],
                "model_largest": round(float(np.log2(said / base_model)), 5)
                if base_model > 0
                else 0.0,
                "unit_largest": round(float(np.log2(answered / base_unit)), 5),
                "model_ms": round(said, 4),
                "unit_ms": round(answered, 4),
                "floor_here": round(floor_here(value, answered), 5),
            }
        )

    over_own = [
        {"value": row["value"], "by": round(abs(row["residual"][0]) - row["floor_here"], 5)}
        for row in rows
        if abs(row["residual"][0]) > row["floor_here"]
    ]
    flat = np.array([abs(row["residual"][0]) for row in rows], dtype=float)
    every = [row["unit_reading"][0] for row in rows]
    span = float(max(every) - min(every)) if every else 0.0
    worst = float(flat.max()) if flat.size else 0.0
    worst_where = next(
        (
            {"value": row["value"], "ms": row["unit_ms"]}
            for row in rows
            if abs(row["residual"][0]) == worst
        ),
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
        "floor_of_the_grid": round(grid_floor, 6),
        "floor_of_one_entry": round(model_floor, 6),
        "is_null_record": span <= 2.0 * floor,
        "settled_by": 0.0,
        # No escape hatch on this path, for the reason the rate path has none: a
        # record here publishes no figure saying its answer was a bound, and
        # inventing one would turn every lean into the measurement's own limit.
        "reading_is_a_value_not_a_bound": True,
        "sign_property": "longer or shorter than the shortest setting the record admits",
        "above": "longer",
        "below": "shorter",
        "readings_not_comparable": not_comparable,
        "over_their_own_floor": over_own,
        "why_over_their_own_floor": WHY_OWN_FLOOR,
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


# ---- a byte as a multiplier

ABOVE_THE_SILENCE_DB = 20.0
"""How far a take has to be above the same chain's silence to be read as a level.

Silence adds to the take rather than replacing it, so a reading this far above it
is high by 0.04 dB, which is what one entry of a table of this size is worth at
the top of the byte where the entries are closest together. At ten decibels it
would be high by 0.41 dB, which is six entries, and the curve would bend at the
bottom for a reason belonging to the room.

A threshold and not a correction. The silence could be subtracted in power and
the excluded settings brought back, and that would be a derivation resting on the
silence takes being the same silence -- taken in the same session, but not at the
same moment as the take they would be subtracted from. The settings it would buy
are the three quietest of a hundred and twenty-eight.
"""


def _level_of(model: dict, address: str, byte_value: int) -> float:
    """What a candidate says the multiplier is at one setting, as a bare ratio.

    Through the same five rules every other map in this module goes through, so a
    candidate that is a formula and one that is a stored table are written in the
    same language and neither gets a shape of its own to be right in.
    """
    return float(_from_map(model["multipliers"][address], byte_value))


def admitted_levels(record: dict) -> tuple[list[dict], list[dict]]:
    """The readings of a level sweep the record's own controls stand behind.

    One exclusion and the record states it itself: a take whose level is not far
    enough above what the same chain recorded with nothing played is a reading of
    the room and the converters, and the figure it returns is the floor's rather
    than the byte's. Which settings those are is read from `above_the_silence_db`,
    which the stage publishes per reading for exactly this.
    """
    kept, left_out = [], []
    for reading in sorted(record.get("readings", []), key=lambda r: int(r["value"])):
        above = reading.get("above_the_silence_db")
        if above is None or float(above) < ABOVE_THE_SILENCE_DB:
            left_out.append(
                {
                    "value": int(reading["value"]),
                    "why": (
                        f"{above} dB above the same chain's silence, and a level is not "
                        f"read below {ABOVE_THE_SILENCE_DB:.0f}"
                    ),
                }
            )
            continue
        kept.append(reading)
    return kept, left_out


PAN_CENTRE = 64
"""The byte the printed range calls 0, which every pan reading is taken against.

Named rather than fitted. The absolute level a take arrives at is the whole
chain's, so a pan can only be read as what it does relative to some setting, and
which setting that is has to come from the printed range rather than from the
numbers -- picking the setting where the two channels came out nearest would be
choosing the reference to make the reference right.
"""


def _pan_sides(model: dict, byte_value: int) -> tuple[float, float]:
    """A pan candidate's two multipliers at one setting of its byte."""
    sides = model["sides"]
    return (
        _from_map({"kind": "points", "log": False, "points": sides["left"]}, byte_value),
        _from_map({"kind": "points", "log": False, "points": sides["right"]}, byte_value),
    )


def _pan_reading(model: dict, byte_value: int, reading: str) -> float:
    left, right = _pan_sides(model, byte_value)
    if reading == "balance":
        return 20.0 * float(np.log10(max(left, 1e-12) / max(right, 1e-12)))
    return 10.0 * float(np.log10(max(left * left + right * right, 1e-24) / 2.0))


def admitted_pans(record: dict, *, run: str) -> tuple[list[dict], list[dict]]:
    """The settings of one saved run that are a reading of the byte, and the rest.

    Two exclusions and both are the record's own. A run the stage could not
    measure has no balance to read; and a setting whose name is not a byte is a
    control the run filed beside its sweep -- the bypassed take, where the part
    was routed past the effect -- which is a reading of the chain rather than of
    the byte, and putting it in the series would have the chain's own imbalance
    scored as a setting.
    """
    found = next((r for r in record.get("runs", []) if r.get("name") == run), None)
    if found is None or not found.get("measured"):
        return [], [{"value": run, "why": found.get("not_measured") if found else "no such run"}]
    kept, left_out = [], []
    for setting in found["by_setting"]:
        name = str(setting["setting"])
        if not name.isdigit():
            left_out.append(
                {
                    "value": name,
                    "why": (
                        "Not a setting of the byte. The run filed its control beside its sweep "
                        "and a control read as a setting is the chain's own imbalance scored as "
                        "one."
                    ),
                }
            )
            continue
        kept.append({**setting, "value": int(name)})
    return sorted(kept, key=lambda s: s["value"]), left_out


def score_against_pans(
    model: dict,
    record: dict,
    *,
    run: str,
    reading: str = "balance",
    floor_by_value: dict[int, float] | None = None,
    separation_ceiling_db: float | None = None,
) -> dict:
    """One saved pan sweep, answered by a candidate pair of multipliers.

    **Two readings and neither is the better one.** The difference between the
    channels is what the byte is printed to move, and three of the candidates in
    this class give the same difference to within a decibel or two. The two
    channels together is what separates them -- a crossfade on the amplitudes is
    three decibels down at the centre, a sine-cosine pair is flat, and a law that
    attenuates only the far side is three decibels up -- and it is the figure a
    comparison made in one channel does not have at all. So a candidate is scored
    under both and claims what survived twice.

    Both sides are taken against PAN_CENTRE, because the level a take arrives at is
    the whole chain's and only a relative reading is the pan's.

    **The floor is per setting, because the reading is.** What bounds a reading of
    a pan is not the take-to-take scatter, which is hundredths of a decibel here;
    it is how far the reading moves when the one thing it must not depend on --
    which voice was sounded -- is changed. That is measured, in the record beside
    this one, and it is three decibels at the ends where it is a tenth in the
    middle. Passing it in is what stops the ends, where every candidate is furthest
    apart, from being read as the sharpest part of the comparison when they are the
    softest.

    **A candidate that predicts silence in a channel is not predicting a number.**
    Every closed form in this class takes its far side to zero at the extreme
    setting, so its balance there is infinite and whatever the arithmetic returns
    is set by the smallest number the code was willing to divide by. That is a
    limit of the writing and it would arrive in the record as a two-hundred-decibel
    residual, which is a figure about a clamp. Where the candidate's separation
    exceeds `separation_ceiling_db` -- what this rig has been shown to resolve, on
    another address, in a record the sweep cites -- the setting leaves the residual
    and becomes a property instead: whether the separation the unit reached is past
    what the rig resolves, which the unit answers and the candidate answers, and
    they either agree or they do not.
    """
    kept, left_out = admitted_pans(record, run=run)
    floors = floor_by_value or {}
    scatter = "balance_spread_db" if reading == "balance" else "together_spread_db"
    at_centre = next((s for s in kept if s["value"] == PAN_CENTRE), None)
    if at_centre is None:
        raise ValueError(
            f"{run} was not asked at {PAN_CENTRE}, which is the byte the printed range calls 0 "
            "and the only setting a pan can be read against without choosing one"
        )

    def measured(setting: dict) -> float:
        values = setting["balance_db"] if reading == "balance" else setting["together_db"]
        return float(np.median(values))

    base_unit = measured(at_centre)
    base_model = _pan_reading(model, PAN_CENTRE, reading)

    rows, per_setting, properties = [], [], []
    ceiling = separation_ceiling_db
    for setting in kept:
        value = setting["value"]
        answered = measured(setting) - base_unit
        predicted = _pan_reading(model, value, reading) - base_model
        if ceiling is not None and abs(_pan_reading(model, value, "balance")) > ceiling:
            reached = abs(float(np.median(setting["balance_db"])))
            properties.append(
                {
                    "value": value,
                    "property": (
                        f"the two channels stand further apart than the {ceiling:.1f} dB this "
                        "chain has been shown to separate"
                    ),
                    "unit": f"no, {reached:.1f} dB",
                    "model": "yes, the far channel is silent",
                    "same": bool(reached > ceiling),
                }
            )
            left_out.append(
                {
                    "value": value,
                    "why": (
                        "The candidate takes its far side to zero here, so the separation it "
                        "predicts is infinite and any residual against it is set by the smallest "
                        f"number the arithmetic would divide by. The unit reached {reached:.1f} "
                        f"dB, against {ceiling:.1f} dB this chain has separated on another "
                        "address, so the disagreement is real and it is stated as a property "
                        "rather than as a number the clamp chose."
                    ),
                }
            )
            continue
        here = max(float(setting.get(scatter) or 0.0), float(floors.get(value, 0.0)))
        per_setting.append(here)
        rows.append(
            {
                "value": value,
                "model_reading": [round(predicted, 4)],
                "unit_reading": [round(answered, 4)],
                "residual": [round(predicted - answered, 4)],
                "model_largest": round(predicted, 4),
                "unit_largest": round(answered, 4),
                "floor_db": round(here, 4),
            }
        )

    floor = float(np.median(per_setting)) if per_setting else 0.0
    flat = np.array([abs(row["residual"][0]) for row in rows], dtype=float)
    every = [row["unit_reading"][0] for row in rows]
    span = float(max(every) - min(every)) if every else 0.0
    worst = float(flat.max()) if flat.size else 0.0
    worst_where = next(
        ({"value": row["value"]} for row in rows if abs(row["residual"][0]) == worst), None
    )
    leans, why_leans = _structured(rows, floor, "dB")
    model_largest = max((abs(row["model_largest"]) for row in rows), default=0.0)
    return {
        "record": None,
        "run": run,
        "reading": reading,
        "measured_in": "dB",
        "span": round(span, 4),
        "floor": round(floor, 4),
        "floor_by_setting": [round(v, 4) for v in per_setting],
        "is_null_record": span <= 2.0 * floor,
        "settled_by": 0.0,
        # The far channel at both ends sits thirty to forty decibels above the
        # take's own lead and the same pair of inputs has separated by fifty-eight
        # on another address, so where this reading stops is the unit's and not a
        # bound the run could not see past.
        "reading_is_a_value_not_a_bound": True,
        "sign_property": (
            "towards the lower-numbered input or the higher"
            if reading == "balance"
            else "louder or quieter than the centre of the byte"
        ),
        "above": "towards the lower-numbered input" if reading == "balance" else "louder",
        "below": "towards the higher-numbered input" if reading == "balance" else "quieter",
        "model_largest": round(model_largest, 4),
        "model_stays_inside_the_floor": bool(model_largest <= floor),
        "median_abs": round(float(np.median(flat)), 4) if flat.size else 0.0,
        "worst_abs": round(worst, 4),
        "worst_at": worst_where,
        "structured": leans,
        "why_structured": why_leans,
        "properties": properties,
        "readings_left_out": left_out,
        "rows": rows,
    }


def score_against_levels(model: dict, record: dict, *, address: str) -> dict:
    """One published level sweep, answered by a candidate multiplier law.

    In decibels relative to the loudest setting, on both sides. **The absolute
    gain is not measurable from this record and the comparison does not pretend
    it is.** What the take carries is the whole chain -- the effect's own
    insertion loss included -- and the run's bypass control says that chain is not
    transparent when its gains are centred: it differs from the flat setting by up
    to four tenths of a decibel. So a law saying the multiplier is the byte over a
    hundred and twenty-seven and one saying it is the byte over a hundred and
    twenty-eight are the same law here, and that is reported as an equivalence
    rather than resolved by picking one.

    The lean is judged against the coarser of what the run resolved and what one
    entry of the candidate's own table is worth, which is admissible because the
    step is measured: every one of the hundred and twenty-eight settings was
    asked, so nothing hides between them.
    """
    kept, left_out = admitted_levels(record)
    run_floor = float(record.get("reference", {}).get("heard_floor_db") or 0.0)
    loudest = max((int(r["value"]) for r in kept), default=127)
    base_unit = next(float(r["heard_db"]) for r in kept if int(r["value"]) == loudest)
    base_model = _level_of(model, address, loudest)

    steps = []
    for value in range(1, 128):
        before = _level_of(model, address, value - 1)
        after = _level_of(model, address, value)
        if after > before > 0:
            steps.append(20.0 * float(np.log10(after / before)))
    model_floor = float(np.median(steps)) if steps else 0.0
    floor = max(run_floor, model_floor)

    rows = []
    for reading in kept:
        value = int(reading["value"])
        said = _level_of(model, address, value)
        answered = float(reading["heard_db"]) - base_unit
        predicted = 20.0 * float(np.log10(max(said, 1e-12) / base_model))
        rows.append(
            {
                "value": value,
                "model_reading": [round(predicted, 4)],
                "unit_reading": [round(answered, 4)],
                "residual": [round(predicted - answered, 4)],
                "model_largest": round(predicted, 4),
                "unit_largest": round(answered, 4),
            }
        )

    flat = np.array([abs(row["residual"][0]) for row in rows], dtype=float)
    every = [row["unit_reading"][0] for row in rows]
    span = float(max(every) - min(every)) if every else 0.0
    worst = float(flat.max()) if flat.size else 0.0
    worst_where = next(
        ({"value": row["value"]} for row in rows if abs(row["residual"][0]) == worst),
        None,
    )
    leans, why_leans = _structured(rows, floor, "dB")
    model_largest = max((abs(row["model_largest"]) for row in rows), default=0.0)
    return {
        "record": None,
        "address": address,
        "measured_in": "dB",
        "span": round(span, 4),
        "floor": round(floor, 4),
        "floor_the_run_resolved": round(run_floor, 4),
        "floor_of_one_entry": round(model_floor, 4),
        "is_null_record": span <= 2.0 * floor,
        "settled_by": 0.0,
        # The same refusal the rate path makes. A level record publishes no figure
        # saying its reading is a bound rather than a value, so there is nothing
        # here that could turn a lean into the measurement's own limit.
        "reading_is_a_value_not_a_bound": True,
        "sign_property": "louder or quieter than the loudest setting of the byte",
        "above": "louder",
        "below": "quieter",
        "model_largest": round(model_largest, 4),
        "model_stays_inside_the_floor": bool(model_largest <= floor),
        "median_abs": round(float(np.median(flat)), 4) if flat.size else 0.0,
        "worst_abs": round(worst, 4),
        "worst_at": worst_where,
        "structured": leans,
        "why_structured": why_leans,
        "readings_left_out": left_out,
        "rows": rows,
    }


def quantum_of(record: dict, *, over: range) -> dict:
    """Which whole-number denominator the measured multipliers land on, if any.

    A gain stored as a fixed-point number lands on a multiple of its own quantum,
    and one computed at the part's word length does not land on anything a sweep
    of this size could see. So the reading is: take every admitted setting's
    multiplier against the loudest, multiply by a candidate denominator, and ask
    how far the results sit from whole numbers. Random reals sit a quarter away on
    average; a denominator that is the real one sits at the measurement's own
    error instead.

    The scan is the control. One denominator tried alone would report a small
    number and there would be nothing to read it against, which is the negative
    with no stated sensitivity this project refuses to publish. What is returned
    is every denominator asked and the runner-up beside the winner.
    """
    kept, left_out = admitted_levels(record)
    loudest = max(int(r["value"]) for r in kept)
    base = next(float(r["heard_db"]) for r in kept if int(r["value"]) == loudest)
    amp = np.array(
        [10.0 ** ((float(r["heard_db"]) - base) / 20.0) for r in kept], dtype=float
    )
    scan = []
    for n in over:
        away = np.abs(amp * n - np.round(amp * n))
        scan.append(
            {
                "denominator": int(n),
                "median_away": round(float(np.median(away)), 4),
                "worst_away": round(float(away.max()), 4),
            }
        )
    ranked = sorted(scan, key=lambda row: row["median_away"])
    return {
        "settings_read": len(kept),
        "settings_left_out": left_out,
        "asked": [int(n) for n in over],
        "winner": ranked[0],
        "runner_up": ranked[1] if len(ranked) > 1 else None,
        "a_random_real_would_be": 0.25,
        "scan": scan,
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
    # Properties the scorer stated for itself. The sign of a deviation is the one
    # this gate can work out from a row unaided; anything else is a property of
    # what was being measured -- which settings returned the same band, which way
    # a reading ran -- and the scorer that knows the unit knows it. The gate stays
    # the arbiter: it does not decide what a property is, only that every one
    # stated has to hold.
    for s in scored:
        for stated in s.get("properties", []):
            if "same" not in stated:
                raise ValueError(
                    f"a property stated by {s.get('record')!r} does not say whether it holds"
                )
            checked.append({"record": s["record"], **stated})
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
    #
    # Leaning is one way to be beaten and not the only one. A candidate that puts
    # the feature in the wrong place at every setting does not lean -- its residual
    # is the same size throughout -- and it is more plainly wrong than one that
    # does. Where the runner-up contradicts a property the winner satisfies, that
    # is the separation, and reading it as no separation is how four candidates two
    # bands apart were once reported as equivalent.
    runner = ranking[1] if len(ranking) > 1 else None
    power_ok = bool(
        runner
        and gross_ok
        and not leaning
        and (
            runner["leaning_records"] > len(leaning)
            or runner.get("contradicts_a_property", False)
        )
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
            "beaten_by": (
                None
                if not runner
                else "leaning where the winner does not"
                if runner["leaning_records"] > len(leaning)
                else "contradicting a property the winner satisfies"
                if runner.get("contradicts_a_property", False)
                else None
            ),
            "separation": (
                round(runner["worst_share_of_span"] - share, 3) if runner else None
            ),
            "why": (
                "The runner-up in the catalogue, scored on the same records, and whether it "
                "leans or contradicts a property where the winner does neither. The decoy is "
                "not chosen -- it is whichever candidate came second, so a strawman cannot be "
                "put up in its place. "
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


RENDERED = ("lti", "table", "pan")
"""The kinds of model this module can hold against the archive.

Named rather than open so that a class nobody has written a renderer for cannot
be scored by accident. A model that claims a *structure* -- the network a reverb
is, the curve a saturating stage bends by, the shape a modulator sweeps in --
would need its output produced sample by sample, and that is not written; one
claiming to be either is refused at the door instead of being fitted with the
wrong instrument.

**What that refusal does not cover is a byte.** A class that asks what quantity a
setting selects is answered by a table against a published curve, whatever the
stage holding that quantity does to a waveform: a delay time is a time whether the
delay saturates, and a modulation rate was identified on twenty-one types without
a single chorus being rendered. The classes still open are almost all of that kind,
and reading them as blocked on a renderer was a misreading of what blocks them --
what each needs is a stage that publishes its quantity, which is a measurement and
not a renderer.


A pan is here because its renderer is written and is two multipliers: nothing has
to be produced for it, since what the archive holds is the level of each channel
and what a candidate says is the level of each channel.
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
