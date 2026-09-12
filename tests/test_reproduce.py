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
        "span_db": span,
        "floor_db": floor,
        "is_null_record": null,
        "settled_by_db": 0.0 if settled else 5.0,
        "profile_settled_inside_the_bands": settled,
        "model_largest_db": model_largest,
        "model_stays_inside_the_floor": model_largest <= floor,
        "median_abs_db": median,
        "worst_abs_db": worst,
        "worst_at": {"value": 0, "hz": 100.0},
        "structured": leans,
        "why_structured": "",
        "bands_above_the_models_nyquist_hz": [],
        "readings_left_out": [],
        "rows": [
            {
                "value": 0,
                "model_band_db": [model_largest],
                "unit_band_db": [unit_largest],
                "residual_db": [median],
                "largest_db": model_largest,
                "unit_largest_db": unit_largest,
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
    assert gates["breakdown"]["records_the_bands_did_not_contain"]


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
        {"residual_db": [0.10, 0.10, 0.10]},
        {"residual_db": [0.12, 0.12, 0.12]},
    ]
    leans, why = reproduce._structured(rows, floor_db=1.0)
    assert not leans and "floor" in why


def test_a_real_lean_is_still_caught():
    rows = [
        {"residual_db": [-2.0, -2.0, -2.0]},
        {"residual_db": [2.0, 2.0, 2.0]},
    ]
    leans, _ = reproduce._structured(rows, floor_db=1.0)
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
