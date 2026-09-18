"""The gates, pointed at what they are supposed to reject.

A suite that only shows the gates passing on the one model that passed them says
nothing: every one of these would also pass if the gate were `return True`. So
each is given a case it must fail, and the cases are the three ways a fit gets
through without being right -- a uniformly poor model riding a large separation,
a strawman decoy, and a residual that leans.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from soundings import reproduce


def scored(
    *,
    span=20.0,
    floor=1.0,
    median=0.5,
    worst=1.0,
    leans=False,
    settled=True,
    null=False,
    unit_largest=10.0,
    model_largest=10.0,
    record="a.json",
) -> dict:
    return {
        "record": record,
        "address": "40 03 04",
        "measured_in": "dB",
        "span": span,
        "floor": floor,
        "is_null_record": null,
        "settled_by": 0.0 if settled else 5.0,
        "reading_is_a_value_not_a_bound": settled,
        "model_largest": model_largest,
        "model_stays_inside_the_floor": model_largest <= floor,
        "median_abs": median,
        "worst_abs": worst,
        "worst_at": {"value": 0, "hz": 100.0},
        "structured": leans,
        "why_structured": "",
        "bands_above_the_models_nyquist_hz": [],
        "readings_left_out": [],
        "rows": [
            {
                "value": 0,
                "model_reading": [model_largest],
                "unit_reading": [unit_largest],
                "residual": [median],
                "model_largest": model_largest,
                "unit_largest": unit_largest,
            }
        ],
    }


def ranking(*leaning_counts: int) -> list[dict]:
    return [
        {"candidate": f"c{i}", "leaning_records": n, "worst_share_of_span": 0.05}
        for i, n in enumerate(leaning_counts)
    ]


def test_a_uniformly_poor_model_fails_the_gross_gate():
    """The hole a decoy-only test leaves open.

    Twelve decibels out on a twenty-five decibel effect is half the effect
    missing, and it can still beat a decoy by a wide margin. The gross gate is
    what stops that, and it is the reason there is a gate on the residual's size
    at all -- as a rejection bound, never as a target.
    """
    gates = reproduce.gates([scored(span=25.0, median=12.0)], ranking=ranking(0, 3))
    assert not gates["gross"]["passed"]
    assert gates["gross"]["residual_over_span"] > reproduce.GROSS_CEILING


def test_a_close_model_passes_the_gross_gate_and_the_number_stops_mattering():
    tight = reproduce.gates([scored(median=0.10)], ranking=ranking(0, 3))
    looser = reproduce.gates([scored(median=0.50)], ranking=ranking(0, 3))
    assert tight["gross"]["passed"] and looser["gross"]["passed"]
    assert tight["power"]["passed"] and looser["power"]["passed"]


def test_a_strawman_decoy_does_not_buy_a_pass():
    """The decoy is whichever candidate came second, and it has to lean.

    Where the runner-up leans on no more records than the winner, the test did
    not separate them and the verdict says so instead of crowning one.
    """
    gates = reproduce.gates([scored()], ranking=ranking(0, 0))
    assert not gates["power"]["passed"]
    assert reproduce.verdict({"gates": gates}, candidates_in_class=2) == (
        "equivalent_under_this_test"
    )


def test_one_candidate_is_a_catalogue_that_is_not_finished():
    gates = reproduce.gates([scored()], ranking=ranking(0))
    assert reproduce.verdict({"gates": gates}, candidates_in_class=1) == "candidates_too_few"


def test_a_leaning_residual_breaks_down():
    gates = reproduce.gates([scored(leans=True)], ranking=ranking(0, 3))
    assert not gates["breakdown"]["passed"]
    assert reproduce.verdict({"gates": gates}, candidates_in_class=3) == "breaks_down"


def test_a_lean_on_a_profile_the_bands_did_not_contain_is_not_a_breakdown():
    """The distinction that keeps a measurement's limit from reading as a defect.

    Where the record's own reading says the profile was still moving at the edge
    of the band set, the corner the model took from it is a bound. A residual
    leaning across those bands is the bound leaning, and no model answers it.
    """
    gates = reproduce.gates([scored(leans=True, settled=False)], ranking=ranking(0, 3))
    assert gates["breakdown"]["passed"]
    assert gates["breakdown"]["records_the_run_did_not_reach"]


def test_a_single_point_past_half_the_span_is_a_breakdown():
    gates = reproduce.gates([scored(span=20.0, median=0.2, worst=11.0)], ranking=ranking(0, 3))
    assert not gates["breakdown"]["passed"]
    assert gates["breakdown"]["breaks_down_at"]


def test_a_model_that_cuts_where_the_unit_boosts_fails_whatever_the_residual():
    gates = reproduce.gates(
        [scored(median=0.01, unit_largest=10.0, model_largest=-10.0)], ranking=ranking(0, 3)
    )
    assert not gates["qualitative"]["passed"]


def test_a_sign_inside_the_floor_is_not_a_sign():
    """Symmetric on purpose: neither side's coin toss counts as a disagreement."""
    gates = reproduce.gates(
        [scored(floor=2.0, unit_largest=1.0, model_largest=-1.0, null=True)],
        ranking=ranking(0, 3),
    )
    assert all(c["same"] for c in gates["qualitative"]["checked"])


def test_a_model_that_invents_an_effect_on_a_null_record_fails():
    gates = reproduce.gates(
        [scored(null=True, floor=1.0, model_largest=6.0)], ranking=ranking(0, 3)
    )
    assert not gates["qualitative"]["passed"]
    assert not gates["gross"]["passed"]


def test_the_lean_is_measured_against_the_floor_and_not_the_residual():
    """A close model must not be held to a tighter lean than a poor one.

    Against its own size, a residual of a tenth of a decibel leans on any
    systematic tenth, so every good model fails. That is what the first pass did.
    """
    rows = [
        {"residual": [0.10, 0.10, 0.10]},
        {"residual": [0.12, 0.12, 0.12]},
    ]
    leans, why = reproduce._structured(rows, 1.0)
    assert not leans and "floor" in why


def test_a_real_lean_is_still_caught():
    rows = [
        {"residual": [-2.0, -2.0, -2.0]},
        {"residual": [2.0, 2.0, 2.0]},
    ]
    leans, _ = reproduce._structured(rows, 1.0)
    assert leans


def test_a_shelf_of_each_order_runs_at_the_slope_its_candidate_predicts():
    """The measurement that separated the two orders, on the model side.

    The catalogue says a first-order shelf runs at about 3.6 dB per octave at
    full gain and a second-order one at about twice that. If the renderer did not
    reproduce that difference, the ranking it produced would be measuring
    something else.
    """
    freq = np.geomspace(20.0, 15000.0, 400)
    slopes = {}
    for order, shelf in ((1, reproduce._first_order_shelf), (2, reproduce._second_order_shelf)):
        db = 20 * np.log10(
            np.abs(shelf(freq, side="low", corner_hz=216.0, gain_db=12.0, fs=32000.0))
        )
        per_octave = np.gradient(db, np.log2(freq))
        slopes[order] = float(np.abs(per_octave).max())
    assert 3.0 < slopes[1] < 4.5, slopes
    assert slopes[2] > 1.7 * slopes[1], slopes


def test_a_byte_past_a_short_table_returns_its_first_entry():
    """Five widths and no bound check, which is what the unit answered.

    Values 5, 64 and 127 all return what 0 returns, and 64 modulo 5 is 4 -- so it
    is an index that fell out of range and not a remainder. A renderer that
    wrapped instead would fit those three settings to the wrong width and the
    residual would blame the shape.
    """
    spec = {"kind": "table", "entries": [0.25, 0.5, 1.0, 2.0, 3.5], "out_of_range": 0}
    assert reproduce._from_map(spec, 4) == 3.5
    for beyond in (5, 64, 127):
        assert reproduce._from_map(spec, beyond) == 0.25


def test_a_gain_byte_sticks_at_both_ends_of_its_window():
    spec = {"kind": "window", "low": 52, "high": 76, "at_low": -12.0, "at_high": 12.0}
    assert reproduce._from_map(spec, 64) == pytest.approx(0.0)
    assert reproduce._from_map(spec, 0) == pytest.approx(-12.0)
    assert reproduce._from_map(spec, 48) == pytest.approx(-12.0)
    assert reproduce._from_map(spec, 127) == pytest.approx(12.0)


def test_nothing_is_returned_above_the_models_own_nyquist():
    """A chain at thirty-two kilohertz has no response at seventeen.

    NaN rather than the value at Nyquist: a number there would be one the model
    never said, and the bands that fall in it are dropped by name instead.
    """
    model = {
        "sample_rate_hz": 32000,
        "chain": [{"kind": "gain", "gain_db": {"fixed": 6.0}}],
    }
    out = reproduce.response(model, {}, np.array([1000.0, 15000.0, 17000.0]))
    assert not np.isnan(out[:2]).any()
    assert np.isnan(out[2])


def test_a_model_that_is_not_time_invariant_is_refused_rather_than_approximated(tmp_path):
    import json

    path = tmp_path / "m.json"
    path.write_text(json.dumps({"model": {"kind": "modulated"}, "chain": []}))
    with pytest.raises(ValueError, match="time-invariant"):
        reproduce.load(path)


# ---------------------------------------------------------------- a byte as a table

TEN = {
    "tables": {
        "0.05 - 10.0": {
            "kind": "steps",
            "first_hz": 0.05,
            "runs": [
                {"through": 99, "step_hz": 0.05},
                {"through": 119, "step_hz": 0.10},
                {"through": 125, "step_hz": 0.50},
            ],
        }
    }
}


def rate_record(pairs, *, agreeing=4, of=4, slowest=0.301, spread=0.004):
    return {
        "address": "40 03 04",
        "readings": [
            {
                "value": v,
                "rate_hz": hz,
                "rates": [hz * (1 - spread), hz * (1 + spread)],
                "agreeing": agreeing,
                "of": of,
                "slowest_measurable_hz": slowest,
            }
            for v, hz in pairs
        ],
    }


def test_a_table_holds_its_last_entry_past_the_end_of_its_runs():
    """One hundred and twenty-six entries, and two byte values with nowhere to go.

    A formula would have run to a hundred and twenty-eight. That the last two
    settings return what the last entry returns is the shape of a list, and it is
    the reason the winning candidate is described as a table at all.
    """
    assert reproduce.rate_of(TEN, "0.05 - 10.0", 99) == pytest.approx(5.00)
    assert reproduce.rate_of(TEN, "0.05 - 10.0", 119) == pytest.approx(7.00)
    assert reproduce.rate_of(TEN, "0.05 - 10.0", 125) == pytest.approx(10.00)
    assert reproduce.rate_of(TEN, "0.05 - 10.0", 127) == pytest.approx(10.00)


def test_a_rate_is_compared_in_octaves_and_not_in_hertz():
    """The half of the range a comparison in hertz would throw away.

    A table right at ten hertz and a tenth of a hertz wrong at the bottom is
    wrong by two octaves there and by nothing a reader would notice at the top.
    In hertz its worst error is a tenth against a span of ten and it passes
    everything.
    """
    record = rate_record([(0, 0.05), (99, 5.0), (127, 10.0)], slowest=0.01)
    good = reproduce.score_against_rates(TEN, record, printed_range="0.05 - 10.0")
    bent = {"tables": {"0.05 - 10.0": dict(TEN["tables"]["0.05 - 10.0"], first_hz=0.15)}}
    bad = reproduce.score_against_rates(bent, record, printed_range="0.05 - 10.0")
    assert good["measured_in"] == "octaves"
    assert good["worst_abs"] < 0.01
    assert bad["worst_abs"] > 1.0
    assert reproduce.gates([bad], ranking=ranking(0, 0))["breakdown"]["passed"] is False


def test_a_table_with_the_wrong_breakpoint_leans_and_is_caught():
    """The one thing only a sweep of the bend can settle.

    Four settings can say a table bends; they cannot say where. A candidate that
    bends one entry early agrees everywhere except across the bend, which is
    exactly a residual that leans.
    """
    unit = [(v, reproduce.rate_of(TEN, "0.05 - 10.0", v)) for v in range(96, 128)]
    early = {
        "tables": {
            "0.05 - 10.0": {
                "kind": "steps",
                "first_hz": 0.05,
                "runs": [
                    {"through": 99, "step_hz": 0.05},
                    {"through": 117, "step_hz": 0.10},
                    {"through": 123, "step_hz": 0.50},
                ],
            }
        }
    }
    scored_here = reproduce.score_against_rates(
        early, rate_record(unit), printed_range="0.05 - 10.0"
    )
    assert scored_here["structured"]
    assert not reproduce.gates([scored_here], ranking=ranking(3, 0))["breakdown"]["passed"]


def test_the_same_readings_do_not_separate_two_candidates_that_agree_on_them():
    """Where the printed range is 6.40, stepping and a straight line are one law.

    A hundred and twenty-eight steps of five hundredths reach six point four
    exactly, so the table and the line through the printed ends are the same
    numbers. Recording that as an equivalence is the answer; picking one would be
    inventing a distinction the readings do not carry.
    """
    stepped = {
        "tables": {
            "0.05 - 6.40": {
                "kind": "steps",
                "first_hz": 0.05,
                "runs": [{"through": 127, "step_hz": 0.05}],
            }
        }
    }
    straight = {"tables": {"0.05 - 6.40": {"kind": "linear", "first_hz": 0.05, "top_hz": 6.40}}}
    for v in (0, 1, 32, 99, 127):
        assert reproduce.rate_of(stepped, "0.05 - 6.40", v) == pytest.approx(
            reproduce.rate_of(straight, "0.05 - 6.40", v)
        )


def test_a_reading_the_partials_did_not_agree_on_is_left_out_with_the_reason():
    record = rate_record([(0, 0.05), (32, 1.65)])
    record["readings"][0]["agreeing"] = 1
    kept, left_out = reproduce.admitted_rates(record)
    assert [r["value"] for r in kept] == [32]
    assert "1 of 4 partials" in left_out[0]["why"]


def test_a_reading_at_the_takes_own_limit_is_the_limit_and_not_a_rate():
    """The stable number that is dangerous because it is stable.

    A take too short to carry two cycles returns its own analysis floor at every
    setting. Read as a rate it says the byte does nothing, which is a claim about
    the unit made out of a property of the recording.
    """
    record = rate_record([(0, 0.30), (32, 1.65)], slowest=0.301)
    kept, left_out = reproduce.admitted_rates(record)
    assert [r["value"] for r in kept] == [32]
    assert "0.301 Hz" in left_out[0]["why"]


def test_a_type_the_page_prints_two_rates_for_is_not_evidence_about_either():
    record = rate_record([(0, 0.05), (127, 10.0)], slowest=0.01)
    ok, why = reproduce.heard_one_modulator(record, rate_slots_printed_for_this_type=4)
    assert not ok and "whichever modulator dominates" in why


def test_a_slot_whose_reading_falls_as_the_byte_rises_is_not_evidence():
    """Not a wrong table -- a take that changed what it was listening to.

    Every candidate in the catalogue agrees that a rate rises with its byte, so
    this test cannot favour one of them. What it separates is a reading of the
    swept modulator from a reading of whatever else the effect was doing.
    """
    record = rate_record([(32, 16.5), (99, 4.9), (127, 4.9)], slowest=0.301)
    ok, why = reproduce.heard_one_modulator(record, rate_slots_printed_for_this_type=1)
    assert not ok and "stopped following one thing" in why


def test_a_sweep_that_only_rises_is_evidence_even_where_it_pauses():
    """Repeated entries at the top are the table, not a take losing the thread."""
    unit = [(v, reproduce.rate_of(TEN, "0.05 - 10.0", v)) for v in range(0, 128)]
    ok, _ = reproduce.heard_one_modulator(
        rate_record(unit, slowest=0.01), rate_slots_printed_for_this_type=1
    )
    assert ok


def _bent(k: float, lo: int = 20, hi: int = 60):
    """The model's own rates, pulled off by `k` entries across the sweep."""
    pairs = []
    for v in range(lo, hi + 1):
        hz = reproduce.rate_of(TEN, "0.05 - 10.0", v)
        entry = np.log2((hz + 0.05) / hz)
        share = (v - lo) / (hi - lo) - 0.5
        pairs.append((v, float(hz / 2 ** (k * entry * share))))
    return pairs


def test_a_lean_smaller_than_one_entry_of_the_table_is_not_a_breakdown():
    """What the floor of one entry is for.

    The run resolves a rate to a few parts in a hundred thousand, which is three
    orders finer than the table's own step. Against that floor every table leans,
    including the one whose entries the readings land on -- so the yardstick has
    to be the coarser of what the run resolved and what a candidate could differ
    by. Below one entry there is no model to write.
    """
    scored_here = reproduce.score_against_rates(
        TEN, rate_record(_bent(0.3), slowest=0.01), printed_range="0.05 - 10.0"
    )
    assert scored_here["floor_of_one_entry"] > scored_here["floor_the_run_resolved"]
    assert not scored_here["structured"]


def test_a_lean_of_several_entries_is_still_caught():
    """And what it is not for. The floor is one entry, not a licence."""
    scored_here = reproduce.score_against_rates(
        TEN, rate_record(_bent(3.0), slowest=0.01), printed_range="0.05 - 10.0"
    )
    assert scored_here["structured"]
    assert not reproduce.gates([scored_here], ranking=ranking(3, 0))["breakdown"]["passed"]


def test_two_units_of_measure_cannot_be_gated_together():
    """A share of a span means nothing across decibels and octaves."""
    band = scored()
    rate = dict(scored(), measured_in="octaves")
    with pytest.raises(ValueError, match="one gate cannot span them"):
        reproduce.gates([band, rate], ranking=ranking(0, 3))


# ---------------------------------------------------------------- a byte as a place

BANDS = [round(1000.0 * 2 ** (k / 12), 1) for k in range(-40, 40)]
"""Twelfth octaves, which is the finest set the equaliser records were read at."""


def band_scored(model_peaks: dict[int, float], unit_peaks: dict[int, float]) -> dict:
    """A band-scored result carrying nothing but where each profile was largest.

    The peak scorer reads only those, so the rest of a real result would be
    scenery. What it must not do is invent a peak where a record has none, and
    the empty rows below are how that is asked.
    """
    return {
        "record": "a.json",
        "address": "40 03 07",
        "measured_in": "dB",
        "band_width_octaves": 1 / 12,
        "span": 12.0,
        "floor": 1.0,
        "is_null_record": False,
        "rows": [
            {
                "value": value,
                "model_peak_hz": model_peaks[value],
                "unit_peak_hz": unit_peaks[value],
                "residual": [0.0],
            }
            for value in sorted(unit_peaks)
        ],
    }


def height_scored(
    model_heights: dict[int, float],
    unit_heights: dict[int, float],
    *,
    floor: float = 0.05,
    skirt: float = 0.0,
) -> dict:
    """A band-scored result carrying a height per setting, and a profile around it.

    `skirt` is how far apart the two profiles are on every band that is not the
    feature's top. It is what the height reading exists to see past: a candidate
    can agree everywhere the feature is small and be wrong where it is not.
    """
    return {
        "record": "a.json",
        "address": "40 03 0C",
        "measured_in": "dB",
        "band_width_octaves": 1 / 12,
        "span": 24.0,
        "floor": floor,
        # A profile whose feature sits at a kilohertz and whose quietest band, where
        # the feature is nothing, scatters twenty times as far.
        "floor_by_band_db": [floor, floor, 20 * floor],
        "bands_hz": [500.0, 1000.0, 12500.0],
        "is_null_record": False,
        "settled_by": 0.0,
        "reading_is_a_value_not_a_bound": True,
        "model_largest": max(abs(v) for v in model_heights.values()),
        "model_stays_inside_the_floor": False,
        "readings_left_out": [],
        "rows": [
            {
                "value": value,
                "model_largest": model_heights[value],
                "unit_largest": unit_heights[value],
                "unit_peak_hz": 1000.0,
                "residual": [skirt] * 20 + [model_heights[value] - unit_heights[value]],
            }
            for value in sorted(unit_heights)
        ],
    }


def test_a_height_is_read_off_both_sides_of_the_same_band_set():
    """A candidate a decibel out at the top is a decibel out, however wide the skirt."""
    here = reproduce.score_against_heights(
        height_scored({52: -11.0, 76: 11.0}, {52: -12.0, 76: 12.0})
    )
    assert here["measured_in"] == "dB"
    assert here["read_as"] == "height"
    assert [abs(r["residual"][0]) for r in here["rows"]] == [1.0, 1.0]
    assert here["median_abs"] == 1.0


def test_a_profile_of_skirts_hides_what_the_height_shows():
    """Why this reading exists, stated as the comparison that goes the wrong way.

    Two candidates against one record. The first is right at the top of the feature
    and slightly out on every band of its skirt; the second is right on the skirt
    and two decibels out where the whole quantity lives. Over the profile the second
    wins on the median, because a profile is mostly skirt. Read as a height it does
    not.
    """
    settings = {52: -12.0, 64: 0.0, 76: 12.0}
    right = height_scored({52: -12.0, 64: 0.0, 76: 12.0}, settings, skirt=0.2)
    wrong = height_scored({52: -10.0, 64: 0.0, 76: 10.0}, settings, skirt=0.0)
    over_the_profile = [
        float(np.median([abs(v) for row in s["rows"] for v in row["residual"]]))
        for s in (right, wrong)
    ]
    assert over_the_profile[0] > over_the_profile[1]
    as_heights = [reproduce.score_against_heights(s)["median_abs"] for s in (right, wrong)]
    assert as_heights[0] < as_heights[1]


def test_a_height_inside_the_runs_own_floor_is_not_a_disagreement():
    here = reproduce.score_against_heights(
        height_scored({52: -12.03, 76: 11.98}, {52: -12.0, 76: 12.0}, floor=0.1)
    )
    assert not here["structured"]
    assert here["worst_abs"] < here["floor"]


def test_a_height_is_bounded_by_the_band_it_was_read_in_and_not_by_the_quietest():
    """The limit of the reading, kept from being written down as the unit's own.

    The record's worst band is twenty times its peak band's, because the feature is
    nothing there and the take sits nearest the floor. Taken as what a height read
    at the peak resolves, a disagreement six times that band's spread disappears.
    """
    scored = height_scored({52: -11.4, 76: 12.0}, {52: -12.0, 76: 12.0}, floor=0.1)
    here = reproduce.score_against_heights(scored)
    assert here["floor"] == 0.1
    assert here["worst_abs"] > here["floor"]
    without = dict(scored)
    without.pop("floor_by_band_db")
    assert reproduce.score_against_heights(without)["floor"] == 0.1
    coarse = dict(scored, floor=2.0)
    coarse.pop("floor_by_band_db")
    assert reproduce.score_against_heights(coarse)["worst_abs"] < 2.0


def stepped(entries, per_entry):
    return {"kind": "stepped-table", "entries": list(entries), "per_entry": per_entry}


def test_a_strided_table_reads_the_same_entry_for_every_byte_in_its_stride():
    spec = stepped([200.0, 250.0, 315.0], 8)
    assert [reproduce._from_map(spec, v) for v in (0, 3, 7)] == [200.0, 200.0, 200.0]
    assert [reproduce._from_map(spec, v) for v in (8, 15)] == [250.0, 250.0]
    assert reproduce._from_map(spec, 16) == 315.0


def test_a_strided_table_holds_its_last_entry_where_the_byte_runs_past_it():
    """Three entries at eight apart reach 23, and the byte reaches 127."""
    spec = stepped([200.0, 250.0, 315.0], 8)
    assert reproduce._from_map(spec, 24) == 315.0
    assert reproduce._from_map(spec, 127) == 315.0


def test_a_place_is_compared_in_octaves_and_not_in_hertz():
    """Two hundred hertz is most of the bottom entry and nothing at the top.

    Both pairs below are out by the same two hundred hertz. In hertz they are one
    error; the reading is a ratio and they are not.
    """
    here = reproduce.score_against_peaks(
        band_scored({0: 400.0, 127: 6300.0}, {0: 200.0, 127: 6100.0})
    )
    assert here["measured_in"] == "octaves"
    low, high = (abs(r["residual"][0]) for r in here["rows"])
    assert low > 0.9 and high < 0.05


def test_a_candidate_with_the_wrong_stride_contradicts_what_the_reading_showed():
    """The failure a residual in decibels cannot see.

    A table of half the stride puts the feature two bands off at settings the
    unit did not tell apart. Its residual stays the same size throughout, so it
    never leans -- and the readings still say plainly that it is wrong.
    """
    unit = {0: 176.8, 4: 176.8, 8: 236.0, 12: 236.0}
    right = reproduce.score_against_peaks(
        band_scored({0: 198.4, 4: 198.4, 8: 250.0, 12: 250.0}, unit)
    )
    wrong = reproduce.score_against_peaks(
        band_scored({0: 198.4, 4: 222.7, 8: 250.0, 12: 280.6}, unit)
    )
    assert all(p["same"] for p in right["properties"])
    missed = [p for p in wrong["properties"] if not p["same"]]
    assert [p["value"] for p in missed] == [4, 12]
    assert missed[0]["unit"] == "not told apart"
    assert missed[0]["model"] == "told apart"


def test_a_candidate_of_the_wrong_stride_the_other_way_is_caught_too():
    """Bracketed, so that the property is not a one-sided test."""
    unit = {0: 176.8, 8: 236.0, 16: 280.6}
    coarse = reproduce.score_against_peaks(band_scored({0: 198.4, 8: 198.4, 16: 333.7}, unit))
    assert [p["value"] for p in coarse["properties"] if not p["same"]] == [8]


def test_two_settings_one_band_apart_are_not_told_apart():
    """The floor and not equality, because the model cannot differ where the unit can.

    A model renders both settings of one entry from the same takes and returns the
    same number twice. The unit's two takes are two takes, and read finely enough
    the largest band moves between them. Read as equality that is the model's
    failure; read against the floor it is what it is.
    """
    here = reproduce.score_against_peaks(band_scored({0: 198.4, 4: 198.4}, {0: 176.8, 4: 187.3}))
    assert here["properties"][0]["unit"] == "not told apart"
    assert here["properties"][0]["same"]


def test_a_property_the_scorer_stated_is_a_gate_and_not_a_note():
    wrong = reproduce.score_against_peaks(band_scored({0: 198.4, 4: 222.7}, {0: 176.8, 4: 176.8}))
    got = reproduce.gates([wrong], ranking=ranking(0, 3))
    assert not got["qualitative"]["passed"]
    assert any(not c["same"] for c in got["qualitative"]["checked"])


def test_a_property_that_does_not_say_whether_it_holds_is_refused():
    """A property nobody answered would pass the gate by not being asked."""
    broken = scored()
    broken["properties"] = [{"property": "something", "unit": "a", "model": "b"}]
    with pytest.raises(ValueError, match="does not say whether it holds"):
        reproduce.gates([broken], ranking=ranking(0, 3))


def test_a_decoy_that_contradicts_a_property_is_a_decoy_that_was_beaten():
    """A candidate can be beaten without leaning, and once was not counted as beaten.

    Putting the feature in the wrong place at every setting is a constant error,
    and a constant error does not lean. Reading that as no separation reported
    four candidates two bands apart as equivalent.
    """
    winner = [scored(leans=False)]
    by_lean = ranking(0, 3)
    assert reproduce.gates(winner, ranking=by_lean)["power"]["passed"]

    neither = [dict(r, leaning_records=0) for r in ranking(0, 0)]
    assert not reproduce.gates(winner, ranking=neither)["power"]["passed"]

    by_property = [dict(r) for r in neither]
    by_property[1]["contradicts_a_property"] = True
    got = reproduce.gates(winner, ranking=by_property)
    assert got["power"]["passed"]
    assert got["power"]["beaten_by"] == "contradicting a property the winner satisfies"


# ---------------------------------------------------------------- a byte as a multiplier


def level_record(levels: dict[int, float], *, floor: float = 0.01, silence: float = -110.0):
    """A level sweep in the shape the stage publishes one."""
    return {
        "address": "40 03 16",
        "reference": {"heard_floor_db": floor},
        "readings": [
            {
                "value": value,
                "heard_db": db,
                "above_the_silence_db": round(db - silence, 1),
            }
            for value, db in sorted(levels.items())
        ],
    }


def level_model(entries):
    return {
        "model": {"kind": "table", "candidate": "a-candidate"},
        "multipliers": {
            "40 03 16": {"kind": "table", "entries": list(entries), "out_of_range": 127}
        },
    }


def test_a_level_is_scored_against_the_loudest_setting_and_not_against_unity():
    """The absolute gain is not in the take, so the comparison does not use one.

    Two candidates a constant factor apart are the same candidate here, which is
    the equivalence this class has to report rather than resolve.
    """
    unit = level_record({v: -53.0 + 20 * np.log10(v / 127) for v in (32, 64, 96, 127)})
    over_127 = reproduce.score_against_levels(
        level_model([v for v in range(128)]), unit, address="40 03 16"
    )
    over_128 = reproduce.score_against_levels(
        level_model([v * 127 / 128 for v in range(128)]), unit, address="40 03 16"
    )
    assert over_127["median_abs"] == pytest.approx(0.0, abs=0.001)
    assert over_128["median_abs"] == pytest.approx(over_127["median_abs"], abs=0.001)


def test_a_law_that_bends_the_wrong_way_leans():
    unit = level_record({v: -53.0 + 20 * np.log10(v / 127) for v in range(8, 128, 8)})
    squared = reproduce.score_against_levels(
        level_model([127 * (v / 127) ** 2 for v in range(128)]), unit, address="40 03 16"
    )
    assert squared["structured"] is True
    assert squared["measured_in"] == "dB"


def test_a_take_too_close_to_the_chains_silence_is_left_out_by_name():
    """A level read at the floor is the room's, and a curve bends there for its sake."""
    unit = level_record({1: -108.0, 8: -77.0, 64: -59.0, 127: -53.0}, silence=-110.0)
    got = reproduce.score_against_levels(level_model(list(range(128))), unit, address="40 03 16")
    assert [r["value"] for r in got["readings_left_out"]] == [1]
    assert [r["value"] for r in got["rows"]] == [8, 64, 127]


def test_the_floor_is_the_coarser_of_the_run_and_one_entry_of_the_table():
    """The two-floor rule, which is admissible because every setting was asked."""
    unit = level_record({v: -53.0 + 20 * np.log10(v / 127) for v in range(8, 128, 8)})
    got = reproduce.score_against_levels(level_model(list(range(128))), unit, address="40 03 16")
    assert got["floor"] == max(got["floor_the_run_resolved"], got["floor_of_one_entry"])
    assert got["floor_of_one_entry"] > got["floor_the_run_resolved"]


def test_a_grid_the_multipliers_sit_on_is_found_and_beats_its_neighbours():
    """The scan is the control: one denominator tried alone has nothing to beat."""
    top = -53.0
    on_a_grid = {v: top + 20 * np.log10(round(127 * (v / 127) ** 1.3) / 127) for v in range(4, 128)}
    on_a_grid[127] = top
    got = reproduce.quantum_of(level_record(on_a_grid), over=range(100, 161))
    assert got["winner"]["denominator"] == 127
    assert got["winner"]["median_away"] < got["runner_up"]["median_away"] / 5


def test_multipliers_on_no_grid_sit_where_a_random_number_would():
    """Otherwise the scan would report a winner whatever it was given."""
    rng = np.random.default_rng(7)
    top = -53.0
    scattered = {v: top + 20 * np.log10(rng.uniform(0.01, 1.0)) for v in range(4, 128)}
    scattered[127] = top
    got = reproduce.quantum_of(level_record(scattered), over=range(100, 161))
    assert got["winner"]["median_away"] > 0.15


# --- a pan, which is two multipliers and two readings ----------------------


def pan_model(law, *, name: str = "a-candidate") -> dict:
    """A candidate as the pair of tables the scorer reads, over the whole byte."""
    return {
        "model": {"schema_version": 1, "id": name, "candidate": name, "kind": "pan"},
        "sides": {
            "left": [[n, law(n)[0]] for n in range(128)],
            "right": [[n, law(n)[1]] for n in range(128)],
        },
    }


def pan_record(
    law,
    *,
    settings=(0, 16, 32, 64, 96, 112, 127),
    run: str = "01-01",
    scatter: float = 0.01,
    extra: list | None = None,
) -> dict:
    """A saved balance run whose two channels follow a law exactly."""
    by = []
    for n in settings:
        left, right = law(n)
        a, b = 20 * np.log10(max(left, 1e-9)), 20 * np.log10(max(right, 1e-9))
        total = 10 * np.log10((left**2 + right**2) / 2 + 1e-18)
        by.append(
            {
                "setting": f"{n:03d}",
                "balance_db": [round(a - b, 4)] * 2,
                "together_db": [round(total, 4)] * 2,
                "balance_spread_db": scatter,
                "together_spread_db": scatter,
            }
        )
    return {"runs": [{"name": run, "measured": True, "by_setting": by + (extra or [])}]}


def sine_cosine(n: int) -> tuple[float, float]:
    return float(np.cos(n * np.pi / 254)), float(np.sin(n * np.pi / 254))


def shallow(n: int) -> tuple[float, float]:
    """A pan that stops short of silence, which is what this unit measured."""
    left, right = sine_cosine(n)
    return left + 0.1 * right, right + 0.1 * left


def test_a_pan_that_follows_a_candidate_exactly_leaves_no_residual():
    """The scorer has to be able to return zero, or a residual it returns means
    nothing. Both readings, because the class claims what survived both."""
    model = pan_model(sine_cosine)
    record = pan_record(sine_cosine)

    for reading in ("balance", "together"):
        got = reproduce.score_against_pans(
            model, record, run="01-01", reading=reading, separation_ceiling_db=58.7
        )
        assert got["median_abs"] < 1e-6, reading


def test_a_candidate_predicting_silence_is_a_property_and_not_a_residual():
    """Every closed form in this class takes its far side to zero at the extreme
    setting, so its balance there is infinite and any residual against it is set
    by the smallest number the arithmetic would divide by -- a two-hundred-decibel
    figure about a clamp. The unit reached 20.6 dB where this chain has separated
    by 58.7, so the disagreement is real and it is stated as a property."""
    model = pan_model(sine_cosine)
    record = pan_record(shallow)

    got = reproduce.score_against_pans(
        model, record, run="01-01", reading="balance", separation_ceiling_db=58.7
    )

    assert [p["value"] for p in got["properties"]] == [0, 127]
    assert all(p["same"] is False for p in got["properties"])
    assert [row["value"] for row in got["rows"]] == [16, 32, 64, 96, 112]
    assert got["worst_abs"] < 20.0


def test_a_control_the_run_filed_beside_its_sweep_is_not_a_setting():
    """The bypassed take is a reading of the chain and not of the byte, and a
    chain with an imbalance of its own scored as a setting is that imbalance
    arriving as a thing the parameter did."""
    record = pan_record(
        sine_cosine,
        extra=[
            {
                "setting": "bypassed",
                "balance_db": [9.0],
                "together_db": [-50.0],
                "balance_spread_db": 0.0,
                "together_spread_db": 0.0,
            }
        ],
    )

    kept, left_out = reproduce.admitted_pans(record, run="01-01")

    assert [s["value"] for s in kept] == [0, 16, 32, 64, 96, 112, 127]
    assert [s["value"] for s in left_out] == ["bypassed"]


def test_the_floor_of_a_pan_is_the_stimulus_and_not_the_scatter():
    """What bounds a reading of a pan is not how well a take repeats -- hundredths
    of a decibel here -- it is how far the reading moves when the one thing it
    must not depend on is changed. Measured, that is three decibels at the ends
    where it is a tenth in the middle, so a single floor would read the ends as
    the sharpest part of the comparison when they are the softest."""
    model = pan_model(sine_cosine)
    record = pan_record(sine_cosine, scatter=0.01)

    got = reproduce.score_against_pans(
        model,
        record,
        run="01-01",
        reading="balance",
        floor_by_value={16: 0.9, 32: 0.4, 64: 0.15, 96: 0.15, 112: 0.9},
        separation_ceiling_db=58.7,
    )

    assert got["floor_by_setting"] == [0.9, 0.4, 0.15, 0.15, 0.9]
    assert got["floor"] == 0.4


def test_a_run_the_stage_could_not_measure_yields_no_readings_and_says_why():
    """A stimulus the run asked and the stage could not answer is absent from a
    scoring rather than scored as nothing, and absent reads as never asked."""
    record = {"runs": [{"name": "01-01", "measured": False, "not_measured": "only one channel"}]}

    kept, left_out = reproduce.admitted_pans(record, run="01-01")

    assert kept == []
    assert left_out == [{"value": "01-01", "why": "only one channel"}]


# ---- a byte as a time


def time_record(pairs, *, step=0.0208333, stands=40.0, admitted=True):
    """A delay record shaped as the stage publishes one.

    `pairs` may name a value twice, which is how a run gives itself a floor, and
    the readings carry their own admission verdict because the stage decides it
    from a bar measured on injected combs rather than from anything a model says.
    """
    return {
        "address": "40 03 03",
        "quefrency_step_ms": step,
        "stands_out": 6.0,
        "readings": [
            {
                "value": v,
                "take": f"t-{i:02d}",
                "ms": ms,
                "stands": stands,
                "admitted": admitted,
            }
            for i, (v, ms) in enumerate(pairs)
        ],
    }


DOUBLING = {
    "tables": {
        "0 - 500m": {
            "kind": "points",
            "log": True,
            "points": [[8, 1.0], [127, 512.0]],
        }
    }
}
"""A byte that doubles the delay every seventeen steps, as a geometric candidate
between two of its own settings. Exact at 8, 25, 42 and so on by construction, so
a test can put a known error in and know it is the only one."""


def test_a_delay_curve_is_compared_in_octaves_and_not_in_milliseconds():
    """Where a byte covers three orders of magnitude, milliseconds hide the bottom.

    A candidate wrong by a factor of three at the short end and right at the long
    one is wrong about the half a player hears as the short setting. In octaves its
    worst error is one and a half and it is rejected; in milliseconds its worst
    error is two against a span of five hundred, and it passes everything.
    """
    unit = [(v, reproduce.time_of(DOUBLING, "0 - 500m", v)) for v in (8, 25, 42, 76, 127)]
    good = reproduce.score_against_times(DOUBLING, time_record(unit), printed_range="0 - 500m")
    bent = {
        "tables": {
            "0 - 500m": dict(DOUBLING["tables"]["0 - 500m"], points=[[8, 3.0], [127, 512.0]])
        }
    }
    bad = reproduce.score_against_times(bent, time_record(unit), printed_range="0 - 500m")
    assert good["measured_in"] == "octaves"
    assert good["worst_abs"] < 0.01
    assert bad["worst_abs"] > 1.0
    assert reproduce.gates([bad], ranking=ranking(0, 0))["breakdown"]["passed"] is False


def test_a_setting_whose_peak_was_the_roughness_is_left_out_with_the_reason():
    """The verdict is the record's, and a model never gets to make it.

    A cepstrum always has a strongest peak, so a setting with no copy in its output
    returns a short, plausible, repeatable time. Scored as a delay it says the byte
    does something at the bottom of its range, which is a claim about the unit made
    out of the roughness of a stimulus.
    """
    record = time_record([(0, 0.46), (64, 14.0)])
    record["readings"][0]["admitted"] = False
    record["readings"][0]["stands"] = 2.4
    kept, left_out = reproduce.admitted_times(record)
    assert [r["value"] for r in kept] == [64]
    assert "2.4 times the carrier's own roughness" in left_out[0]["why"]


def test_the_grid_floors_a_run_whose_repeats_came_back_identical():
    """Two takes landing in one cell is the grid, and not the reading being exact.

    The answer is quantised to one quefrency of the transform it was read through,
    so repeats can agree to the last digit however rough the takes were. A floor
    drawn from those repeats alone is zero, and against zero every candidate leans.
    """
    same = [(64, 14.0), (64, 14.0), (8, 1.0), (127, 512.0)]
    run_floor, grid_floor = reproduce._time_floors(
        time_record(same), time_record(same)["readings"]
    )
    assert run_floor == 0.0
    assert grid_floor > 0.0

    scored_here = reproduce.score_against_times(
        DOUBLING, time_record(same), printed_range="0 - 500m"
    )
    assert scored_here["floor_the_run_resolved"] == 0.0
    assert scored_here["floor"] >= scored_here["floor_of_the_grid"] > 0.0


def test_the_grid_is_most_of_a_short_delay_and_nothing_of_a_long_one():
    """Which is why the floor is taken in octaves rather than in milliseconds.

    One quefrency is a fiftieth of a millisecond. Against a copy at one millisecond
    that is a real part of the answer; against one at half a second it is nothing,
    and a floor stated in milliseconds would be the same number for both.
    """
    short = time_record([(1, 0.5), (2, 0.6)])
    long_one = time_record([(120, 400.0), (127, 500.0)])
    assert reproduce._time_floors(short, short["readings"])[1] > 0.02
    assert reproduce._time_floors(long_one, long_one["readings"])[1] < 0.001


def test_a_candidate_that_bends_one_entry_early_leans_and_is_caught():
    """The one thing only a sweep of every setting can settle.

    A handful of settings can say a table bends; they cannot say where. A candidate
    bending early agrees everywhere except across the bend, which is a residual
    that leans -- and the floor of one entry is only admissible as a floor because
    the slot behind this class was asked at all of its settings.
    """
    kinked = {
        "tables": {
            "0 - 500m": {
                "kind": "points",
                "log": True,
                "points": [[8, 1.0], [64, 16.0], [127, 512.0]],
            }
        }
    }
    early = {
        "tables": {
            "0 - 500m": {
                "kind": "points",
                "log": True,
                "points": [[8, 1.0], [48, 16.0], [127, 512.0]],
            }
        }
    }
    unit = [(v, reproduce.time_of(kinked, "0 - 500m", v)) for v in range(8, 128, 4)]
    scored_here = reproduce.score_against_times(
        early, time_record(unit), printed_range="0 - 500m"
    )
    assert scored_here["structured"]
    assert not reproduce.gates([scored_here], ranking=ranking(3, 0))["breakdown"]["passed"]


def test_a_setting_a_candidate_puts_no_time_on_is_named_and_not_passed_over() -> None:
    """A residual is a ratio and neither side of it can be nothing.

    The printed range of a delay begins at zero, so a candidate that follows the
    page says the bottom setting is no delay at all. There is no octave to be wrong
    by there, and the row has to leave the comparison -- but leaving it silently is
    a shorter sweep published as a whole one, which is the shape of drop this
    archive refuses everywhere else.
    """
    at_zero = {
        "tables": {"0 - 500m": {"kind": "points", "log": False, "points": [[0, 0.0], [127, 500.0]]}}
    }
    got = reproduce.score_against_times(
        at_zero, time_record([(0, 0.1042), (64, 252.0), (127, 500.0)]), printed_range="0 - 500m"
    )
    named = [row["value"] for row in got["readings_not_comparable"]]
    assert named == [0]
    assert got["readings_not_comparable"][0]["why"]
    assert 0 not in [row["value"] for row in got["rows"]]


def test_a_setting_is_judged_against_its_own_resolution_and_not_the_run_s_median() -> None:
    """One floor cannot be the floor of a byte covering three orders of magnitude.

    Half a quefrency is most of an octave at a tenth of a millisecond and nothing at
    half a second, so which floor a setting has is a property of that setting. A
    candidate out by a tenth of an octave at the long end is plainly out; the same
    tenth at the shortest setting is inside what the grid could have returned. Judged
    against one number over the whole byte both are called the same way, and one of
    those calls is wrong whichever way the number falls.
    """
    ends = {
        "tables": {
            "0 - 500m": {"kind": "points", "log": True, "points": [[0, 0.1], [127, 500.0]]}
        }
    }
    asked = [0, 40, 80, 127]
    said = {v: reproduce.time_of(ends, "0 - 500m", v) for v in asked}

    def moved(at: int) -> list[tuple[int, float]]:
        # A tenth of an octave: under half a quefrency at a tenth of a millisecond,
        # and a hundred quefrencies at half a second.
        return [(v, said[v] * (2**-0.13 if v == at else 1.0)) for v in asked]

    exact = reproduce.score_against_times(
        ends, time_record([(v, said[v]) for v in asked]), printed_range="0 - 500m"
    )
    assert exact["over_their_own_floor"] == []

    short = reproduce.score_against_times(
        ends, time_record(moved(0)), printed_range="0 - 500m"
    )
    long = reproduce.score_against_times(
        ends, time_record(moved(127)), printed_range="0 - 500m"
    )
    assert short["over_their_own_floor"] == []
    assert [row["value"] for row in long["over_their_own_floor"]] == [127]
    # Both were moved by the same number of octaves, and the one number the run
    # would otherwise be judged by does not tell them apart.
    assert short["worst_abs"] == pytest.approx(long["worst_abs"], abs=1e-4)


def test_a_long_delay_is_judged_against_two_clocks_and_a_short_one_is_not() -> None:
    """The one term in the floor that is a scale error rather than a quantisation.

    A delay is counted in the unit's clock and read in the one the take was captured
    on, so every reading is scaled by whatever those two sit apart. That is nothing
    where the delay is short and more than a quefrency where it is a second, and a
    floor built only from the grid and the ladder's own entries is therefore tighter
    than the rig at the long end -- which shows up as the top of a saturated range
    disagreeing by a fraction of one quefrency.

    It must not become a floor that forgives anything. A candidate wrong by far more
    than the two clocks could account for still fails with the term in.
    """
    flat_top = {
        "tables": {
            "0 - 500m": {
                "kind": "points",
                "log": False,
                "points": [[0, 0.0], [120, 1000.0], [127, 1000.0]],
            }
        }
    }
    # The unit reads the plateau 20 ppm long, which is inside the rig and outside
    # half a quefrency at that time.
    plateau = [(v, 1000.0 * (1 + 20e-6)) for v in range(121, 128)]
    # A short setting the candidate is exactly right at, so the run has something
    # in it the clock cannot reach and the floor is not read off the plateau alone.
    record = time_record([(8, reproduce.time_of(flat_top, "0 - 500m", 8)), *plateau])

    tight = reproduce.score_against_times(flat_top, record, printed_range="0 - 500m")
    assert [r["value"] for r in tight["over_their_own_floor"]] == list(range(121, 128))

    allowed = reproduce.score_against_times(
        flat_top, record, printed_range="0 - 500m", clock_ppm=57.0
    )
    assert allowed["over_their_own_floor"] == []
    assert allowed["clock_ppm_allowed_for"] == 57.0

    # The same term, against a candidate wrong by a whole step rather than by ppm.
    bent = {
        "tables": {
            "0 - 500m": {
                "kind": "points",
                "log": False,
                "points": [[0, 0.0], [120, 800.0], [127, 800.0]],
            }
        }
    }
    still_fails = reproduce.score_against_times(
        bent, record, printed_range="0 - 500m", clock_ppm=57.0
    )
    over = {r["value"] for r in still_fails["over_their_own_floor"]}
    assert set(range(121, 128)) <= over, (
        "the clock term must not forgive a candidate wrong by a whole step"
    )


def test_the_clock_term_is_not_a_constant_beside_the_code() -> None:
    """A figure measured on one unit is not another unit's default.

    The drift is a property of one machine against one interface. Asked for without
    one, the scorer allows nothing for it and says so, so a caller that forgot is
    tighter than the rig rather than quietly carrying somebody else's number.
    """
    record = time_record([(8, 32.0), (64, 256.0)])
    got = reproduce.score_against_times(DOUBLING, record, printed_range="0 - 500m")
    assert got["clock_ppm_allowed_for"] is None

    asked_for = inspect.signature(reproduce.score_against_times).parameters["clock_ppm"]
    assert asked_for.default is None, (
        "a default here would be one unit's measurement arriving as another's "
        "assumption, which is the one thing no stage in this repo may do"
    )


def _q_at_the_midpoint(transfer, freq) -> tuple[float, float]:
    """How tall a section stands and how wide it is between its half-gain points.

    The width every filter cookbook defines its own width parameter at, read off a
    rendering rather than taken from the coefficients -- so that a section reached
    two ways is measured the same way both times.
    """
    db = 20 * np.log10(np.abs(transfer))
    top = int(np.argmax(np.abs(db)))
    half = db[top] / 2.0
    edges = []
    for step in (-1, 1):
        j = top
        while 0 < j < len(freq) - 1 and (db[j] - half) * (db[j + step] - half) > 0:
            j += step
        edges.append(freq[j])
    low, high = sorted(edges)
    return float(db[top]), float(np.sqrt(low * high) / (high - low))


def _a_peak(freq):
    from functools import partial

    return partial(reproduce._peaking, freq, centre_hz=1250.0, q=1.0, fs=32000.0)


def test_a_stage_that_says_nothing_about_its_gain_is_built_at_it():
    """The path every model in the archive was written against, unchanged.

    A renderer that grew a second way of reaching a section and quietly moved the
    first onto it would restate every closed claim without anything failing.
    """
    freq = np.geomspace(60.0, 15000.0, 6000)
    for gain in (-12.0, -6.0, 0.0, 6.0, 12.0):
        direct = reproduce._peaking(freq, centre_hz=1250.0, q=1.0, gain_db=gain, fs=32000.0)
        through = reproduce._how_the_gain_reaches({"kind": "peaking"}, _a_peak(freq), gain)
        assert np.allclose(direct, through)


def test_a_mixed_section_stands_where_the_byte_asked_at_every_setting():
    """The mix is solved and not guessed.

    A blend whose mix runs with the byte rather than with what the byte asks for
    would be a different gain law as well as a different shape, and the residual
    could not say which of the two it was reading.
    """
    freq = np.geomspace(60.0, 15000.0, 6000)
    for stage in (
        {"kind": "peaking", "reached_by": {"full_db": 12.0}},
        {
            "kind": "peaking",
            "reached_by": {
                "full_db": 12.0,
                "cut": "towards-a-second-section-stored-at-full-cut",
            },
        },
    ):
        for gain in (-12.0, -6.0, -1.0, 1.0, 6.0, 12.0):
            top, _ = _q_at_the_midpoint(
                reproduce._how_the_gain_reaches(stage, _a_peak(freq), gain), freq
            )
            assert top == pytest.approx(gain, abs=0.02), (stage, gain, top)


def test_the_two_ways_of_reaching_a_section_agree_at_the_ends_and_not_between():
    """What this class is scored on, on the model side.

    At the top of the byte's range the blend is the whole section and the two
    readings are one curve; at the middle of it they are not, and by more than any
    run's floor. A renderer that returned the same width at half gain would make
    the class untestable while every gate went on passing.
    """
    freq = np.geomspace(60.0, 15000.0, 6000)
    built = {"kind": "peaking"}
    mixed = {"kind": "peaking", "reached_by": {"full_db": 12.0}}
    at_full = [
        _q_at_the_midpoint(reproduce._how_the_gain_reaches(s, _a_peak(freq), 12.0), freq)[1]
        for s in (built, mixed)
    ]
    assert at_full[0] == pytest.approx(at_full[1], rel=1e-6)
    at_half = [
        _q_at_the_midpoint(reproduce._how_the_gain_reaches(s, _a_peak(freq), 6.0), freq)[1]
        for s in (built, mixed)
    ]
    assert at_half[1] > 1.3 * at_half[0], at_half


def test_an_inverted_cut_mirrors_its_boost_and_a_second_section_does_not():
    """The two cut readings, which is the whole of what separates them.

    Swapping a section's numerator for its denominator makes a cut the boost's
    mirror, so its width at any setting is the width of the boost at that setting.
    Blending towards a second section stored at full cut makes the cut wider, which
    is the asymmetry this reading has to be able to be wrong about.
    """
    freq = np.geomspace(60.0, 15000.0, 6000)
    inverted = {"kind": "peaking", "reached_by": {"full_db": 12.0}}
    second = {
        "kind": "peaking",
        "reached_by": {
            "full_db": 12.0,
            "cut": "towards-a-second-section-stored-at-full-cut",
        },
    }
    widths = {}
    for name, stage in (("inverted", inverted), ("second", second)):
        widths[name] = {
            gain: _q_at_the_midpoint(
                reproduce._how_the_gain_reaches(stage, _a_peak(freq), gain), freq
            )[1]
            for gain in (6.0, -6.0)
        }
    assert widths["inverted"][-6.0] == pytest.approx(widths["inverted"][6.0], rel=1e-3)
    assert widths["second"][-6.0] < 0.6 * widths["second"][6.0], widths
