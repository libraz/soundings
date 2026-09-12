"""The gates, pointed at what they are supposed to reject.

A suite that only shows the gates passing on the one model that passed them says
nothing: every one of these would also pass if the gate were `return True`. So
each is given a case it must fail, and the cases are the three ways a fit gets
through without being right -- a uniformly poor model riding a large separation,
a strawman decoy, and a residual that leans.
"""

from __future__ import annotations

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
    bent = {
        "tables": {
            "0.05 - 10.0": dict(TEN["tables"]["0.05 - 10.0"], first_hz=0.15)
        }
    }
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
    straight = {
        "tables": {"0.05 - 6.40": {"kind": "linear", "first_hz": 0.05, "top_hz": 6.40}}
    }
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
    coarse = reproduce.score_against_peaks(
        band_scored({0: 198.4, 8: 198.4, 16: 333.7}, unit)
    )
    assert [p["value"] for p in coarse["properties"] if not p["same"]] == [8]


def test_two_settings_one_band_apart_are_not_told_apart():
    """The floor and not equality, because the model cannot differ where the unit can.

    A model renders both settings of one entry from the same takes and returns the
    same number twice. The unit's two takes are two takes, and read finely enough
    the largest band moves between them. Read as equality that is the model's
    failure; read against the floor it is what it is.
    """
    here = reproduce.score_against_peaks(
        band_scored({0: 198.4, 4: 198.4}, {0: 176.8, 4: 187.3})
    )
    assert here["properties"][0]["unit"] == "not told apart"
    assert here["properties"][0]["same"]


def test_a_property_the_scorer_stated_is_a_gate_and_not_a_note():
    wrong = reproduce.score_against_peaks(
        band_scored({0: 198.4, 4: 222.7}, {0: 176.8, 4: 176.8})
    )
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
