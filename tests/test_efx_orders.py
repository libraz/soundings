"""Controls for reading a setting as harmonic orders rather than as a level.

Every failure this guards against publishes rather than raising. A reading that
moves when a gain after the stage moves is not a fingerprint of a curve, and
nothing in the figures would say so -- it would read as the setting having changed
the distortion. A carrier taken from the note rather than from the take reads every
order through the skirt of its own filter, and returns smaller orders at every
setting, which reads as a gentler stage. A record that names both an address and a
controller reads as a parameter somebody swept.

The takes here are synthetic and put through a polynomial, so the orders are known
in closed form: a curve whose cube term is a known size has to come back with the
third order that term produces, and a reading that returns most of it is a reading
with a leak in it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from soundings import efxorders, index, partials, takes
from soundings.cli import blocks

SR = 48000
SECONDS = 3.0
CARRIER = 200.0

SQUARED, CUBED = 0.2, 0.4
"""The curve every take below is put through: `x + 0.2 x^2 + 0.4 x^3`.

Both terms on purpose. A curve with only an odd term returns only odd orders, and
a reading that dropped every even one would pass against it."""

FIRST = 1.0 + 0.75 * CUBED
SECOND = SQUARED / 2.0
THIRD = CUBED / 4.0
"""What that curve makes of a unit sine, from the expansions of its two terms:
`x^2 = (1 - cos 2x)/2` and `x^3 = (3 sin x - sin 3x)/4`."""

EXPECTED = (
    round(float(20.0 * np.log10(SECOND / FIRST)), 3),
    round(float(20.0 * np.log10(THIRD / FIRST)), 3),
)


@dataclass
class FakeRecording:
    samples: np.ndarray
    sample_rate: int = SR
    device: str = "test input"
    overflows: int = 0
    open_seconds: float = 0.6

    @property
    def seconds(self) -> float:
        return self.samples.shape[0] / self.sample_rate


def through_the_curve(
    *, hz: float = CARRIER, gain: float = 1.0, seconds: float = SECONDS, curved: bool = True
) -> np.ndarray:
    """A sine through the polynomial, then scaled -- which is the order of the test.

    The gain is applied after the curve and never before it, because those are two
    different questions and only one of them is what a fingerprint is for: a gain
    before the curve moves the orders, a gain after it must not.
    """
    at = np.arange(int(seconds * SR), dtype=np.float64) / SR
    body = np.sin(2 * np.pi * hz * at)
    if curved:
        body = body + SQUARED * body**2 + CUBED * body**3
    body = body * 0.2 * gain
    return np.stack([body, body * 0.5], axis=1)


def a_run(where, *, takes_at, seconds: float = SECONDS):
    """A directory of takes named the way a sweep names them."""
    store = takes.Store.open(where)
    for label, samples in takes_at:
        store.keep(FakeRecording(samples), stimulus="held", setting=label, take=0)
    store.close(question="a run of a known curve")
    return where


def read(where, **extra):
    return efxorders.read_directory(
        where,
        type_id="01 10",
        setting=r"held-drive-(?P<value>\d+)-00",
        carrier_hz=CARRIER,
        hold_s=SECONDS - 1.0,
        **{"address": "40 03 03", "control": r"held-bypassed-\d+-00", **extra},
    )


@pytest.fixture
def one_curve(tmp_path):
    """One curve at three settings, each at a different loudness after the stage.

    The three are the same take through the same polynomial and differ only in what
    was done to the output afterwards, so a reading of the curve has to return one
    answer three times.
    """
    return a_run(
        tmp_path / "curve",
        takes_at=[
            *[(f"flat-{i:02d}", through_the_curve()) for i in range(3)],
            ("drive-000", through_the_curve()),
            ("drive-064", through_the_curve(gain=10.0)),
            ("drive-127", through_the_curve(gain=0.05)),
            ("bypassed-00", through_the_curve(curved=False)),
        ],
    )


def test_a_known_curve_comes_back_with_the_orders_it_makes(one_curve) -> None:
    """The reading against a closed form rather than against another reading."""
    found = read(one_curve)
    first = next(row for row in found["readings"] if row["value"] == 0)
    assert first["under_the_first_db"][:2] == pytest.approx(EXPECTED, abs=0.05)
    # The curve has no term above the third, so nothing above the third order is
    # the stage's. Far below rather than exactly nothing: what is there is the
    # arithmetic's own residue, and a bound is what says the reading is not
    # inventing orders.
    assert max(first["under_the_first_db"][2:]) < -100.0


def test_a_gain_after_the_stage_does_not_move_the_orders(one_curve) -> None:
    """The property the whole reading rests on, and the one a level cannot have.

    Three takes of one curve, twenty-six decibels apart in what reached the
    converter. A reading that moved with them would report a gain as a change in
    the distortion, and every comparison between two settings of different
    loudness would be a comparison through it.
    """
    found = read(one_curve)
    rows = {row["value"]: row["under_the_first_db"] for row in found["readings"]}
    # The two orders this curve actually makes. The ones above them are the
    # arithmetic's own residue two hundred decibels down, and scaling a take moves
    # that against the float floor -- which is a fact about the test's signal and
    # not about the reading.
    spread = [max(abs(rows[0][i] - rows[value][i]) for value in rows) for i in (0, 1)]
    assert max(spread) < 0.01
    # And the levels really were far apart, or the test above passes on takes that
    # were never a test of it.
    heard = [row["heard_db"] for row in found["readings"]]
    assert max(heard) - min(heard) > 20.0


def test_the_takes_routed_past_the_stage_carry_none_of_its_orders(one_curve) -> None:
    """Without the control every order in the record implies a stage that made it."""
    found = read(one_curve)
    assert found["control"]["takes"] == ["held-bypassed-00-00.wav"]
    assert max(found["control"]["readings"][0]["under_the_first_db"]) < -100.0


def test_the_carrier_is_measured_and_not_taken_from_what_was_asked_for(tmp_path) -> None:
    """A unit's own tuning moves the carrier, and a filter aimed off it reads low.

    Three hertz off two hundred is well inside what a tuning offset reaches and
    far more than the reading's own resolution. Asked at the nominal and found at
    the real one, the orders have to come back at the same figures as a take that
    was never detuned.
    """
    where = a_run(
        tmp_path / "detuned",
        takes_at=[
            ("flat-00", through_the_curve(hz=CARRIER + 3.0)),
            ("drive-000", through_the_curve(hz=CARRIER + 3.0)),
        ],
    )
    found = read(where, reference=r"held-flat-\d+-00")
    row = next(r for r in found["readings"] if r["value"] == 0)
    assert row["fundamental_hz"] == pytest.approx(CARRIER + 3.0, abs=0.05)
    assert row["under_the_first_db"][:2] == pytest.approx(EXPECTED, abs=0.05)


def test_a_carrier_taken_from_the_note_instead_reads_the_orders_smaller(tmp_path) -> None:
    """What the test above is protecting against, shown to be worth protecting.

    The same detuned take read with the measurement disabled -- which is what a
    reading that trusted the note number would do. It does not fail; it returns a
    gentler stage.
    """
    at = np.arange(int(SECONDS * SR), dtype=np.float64) / SR
    body = np.sin(2 * np.pi * (CARRIER + 3.0) * at)
    body = body + SQUARED * body**2 + CUBED * body**3
    aimed = partials.levels(body, SR, carrier_hz=CARRIER, count=4)
    right = partials.levels(body, SR, carrier_hz=CARRIER + 3.0, count=4)
    assert efxorders.under_the_first(right)[:2] == pytest.approx(EXPECTED, abs=0.05)
    assert efxorders.under_the_first(aimed)[1] < EXPECTED[1] - 3.0


def test_a_record_is_about_one_address_or_one_controller(one_curve) -> None:
    """Both or neither would leave a reader to guess which the figures belong to."""
    with pytest.raises(ValueError):
        read(one_curve, controller=7)
    with pytest.raises(ValueError):
        efxorders.read_directory(
            one_curve,
            type_id="01 10",
            setting=r"held-drive-(?P<value>\d+)-00",
            carrier_hz=CARRIER,
            hold_s=SECONDS - 1.0,
        )


def test_a_controller_subject_lands_in_the_field_the_listing_groups_on(one_curve) -> None:
    """A run that swept a controller has to meet the records that sent one.

    Filed under its own spelling it would be a group of one, which reads as a
    subject nothing else in the archive has ever been asked about.
    """
    found = read(one_curve, address=None, controller=7)
    assert "address" not in found
    # The grouping key rather than the fields, because that is what puts two
    # records about one thing in one place.
    assert index.as_a_subject(index.about(found)) == "controller 7, type 01 10"


def test_a_run_with_no_repeats_in_it_still_reads_and_says_it_has_no_floor(
    one_curve,
) -> None:
    """The difference from the band reading beside this one, held rather than stated.

    An order is under its own take's first order, so the figures do not need the
    repeats. What the repeats give is the floor, and a record without them has to
    say that rather than carry a floor of nothing -- which every difference would
    clear.
    """
    found = read(one_curve, reference=None)
    assert found["reference"]["floor_db"] is None
    assert found["readings"]
    assert all("largest_db" not in row for row in found["readings"])


def test_a_difference_inside_the_repeats_own_spread_is_not_a_reading(one_curve) -> None:
    """The setting the reference was made of, asked as a setting.

    It is the same state, so every order has to fall inside the floor the repeats
    drew. A reading that came back with a largest order here would be reporting the
    run's own scatter as something the byte did.
    """
    found = read(one_curve, reference=r"held-flat-\d+-00")
    first = next(row for row in found["readings"] if row["value"] == 0)
    assert first["largest_db"] is None
    assert first["outside_the_floor_orders"] == []


def test_the_repeats_are_claimed_before_the_sweep_can_match_them(tmp_path) -> None:
    """A sweep pattern wide enough to name the repeats folds them into the readings.

    They are the same state, so the curve grows points that never moved and still
    looks like a curve.
    """
    where = a_run(
        tmp_path / "overlapping",
        takes_at=[
            ("drive-000", through_the_curve()),
            ("drive-064", through_the_curve(gain=4.0)),
        ],
    )
    found = efxorders.read_directory(
        where,
        type_id="01 10",
        address="40 03 03",
        setting=r"held-drive-(?P<value>\d+)-00",
        reference=r"held-drive-000-00",
        carrier_hz=CARRIER,
        hold_s=SECONDS - 1.0,
    )
    assert [row["value"] for row in found["readings"]] == [64]


def test_a_control_made_at_every_setting_says_which_setting_each_was(tmp_path) -> None:
    """A run that swept something in front of the effect has a ladder for a control.

    What the stage was given is not in the type and nothing inside it reads its own
    input, so the bypassed takes are the only measurement of it -- and a ladder
    whose rungs are unlabelled is not one. Where the pattern captures the setting it
    is kept; where it does not, the control is one state and carries no value.
    """
    where = a_run(
        tmp_path / "in-front",
        takes_at=[
            ("flat-00", through_the_curve()),
            ("drive-000", through_the_curve()),
            ("bypassed-000", through_the_curve(curved=False)),
            ("bypassed-064", through_the_curve(curved=False, gain=0.5)),
        ],
    )
    labelled = read(
        where,
        reference=r"held-flat-\d+-00",
        control=r"held-bypassed-(?P<value>\d+)-00",
    )
    assert [row["value"] for row in labelled["control"]["readings"]] == [0, 64]
    plain = read(where, reference=r"held-flat-\d+-00")
    assert all("value" not in row for row in plain["control"]["readings"])


def test_a_short_filter_reads_what_sits_beside_an_order_as_part_of_it() -> None:
    """Why the reading runs over the whole stretch and what the other width is for.

    A tone put between the second and third orders belongs to neither. A boxcar one
    carrier period long has a lobe a whole fundamental wide, so it collects that
    tone into the order beside it and returns a stage that made more than it did;
    the filter over the whole stretch is narrow enough to leave it where it is.
    Both are readings of the same take, which is why the width a record was read at
    is in the record.
    """
    at = np.arange(int(SECONDS * SR), dtype=np.float64) / SR
    body = np.sin(2 * np.pi * CARRIER * at)
    body = body + SQUARED * body**2 + CUBED * body**3
    # Midway between the two orders and twice the size of the smaller of them, so
    # that what each filter does with it is unmistakable rather than marginal.
    beside = body + 2.0 * THIRD * np.sin(2 * np.pi * CARRIER * 2.5 * at)
    narrow = efxorders.under_the_first(partials.levels(beside, SR, carrier_hz=CARRIER, count=3))
    wide = efxorders.under_the_first(
        partials.levels(beside, SR, carrier_hz=CARRIER, count=3, periods=1)
    )
    assert narrow[:2] == pytest.approx(EXPECTED, abs=0.05)
    assert min(wide[0] - narrow[0], wide[1] - narrow[1]) > 3.0


def test_the_count_of_orders_the_parser_offers_is_the_one_the_reader_holds() -> None:
    """Spelled twice so the parser can be built without loading the reader."""
    assert blocks.ORDERS_READ == efxorders.ORDERS


def test_the_series_says_how_much_level_it_still_carried_at_the_last_order(
    one_curve,
) -> None:
    """Otherwise the total below is read as the whole of what the stage returned."""
    found = read(one_curve)
    first = next(row for row in found["readings"] if row["value"] == 0)
    # This curve has nothing above its third order, so the series has fallen away
    # long before the count runs out.
    assert first["settled_at_the_top_db"] < -80.0


def turns_of(body: np.ndarray) -> list[float]:
    """The turn of each order of one body, through the reader's own routine."""
    return efxorders._turns(body, SR, CARRIER, 4)


def a_curve(*, hz: float = CARRIER, seconds: float = SECONDS) -> np.ndarray:
    at = np.arange(int(seconds * SR), dtype=np.float64) / SR
    body = np.sin(2 * np.pi * hz * at)
    return body + SQUARED * body**2 + CUBED * body**3


def test_a_stage_with_no_state_in_it_turns_no_order_off_the_carrier() -> None:
    """The reading that separates a curve from something with a memory.

    A memoryless curve fed one steady tone can only return each order along the
    carrier or exactly against it. Anything between is not a curve, and a reading of
    sizes cannot see the difference at all -- which is the whole reason the turns are
    in the record beside the sizes.
    """
    turns = turns_of(a_curve())
    for turn in turns[:3]:
        assert min(abs(turn), abs(abs(turn) - 180.0)) < 0.5


def test_a_filter_after_the_curve_shows_in_the_turns() -> None:
    """And the sizes it moves do not say it was a filter rather than a curve.

    A one-pole whose corner sits among the orders turns each of them by its own angle
    there, which is not the same angle times the order, so nothing that removes a
    delay can absorb it. This is the case the turns are a detector for.
    """
    body = a_curve()
    keep = float(np.exp(-2 * np.pi * (CARRIER * 1.5) / SR))
    after = np.zeros_like(body)
    for i in range(1, after.size):
        after[i] = keep * after[i - 1] + (1.0 - keep) * body[i]
    assert max(abs(turn) for turn in turns_of(after)[1:3]) > 10.0


def test_the_turns_do_not_depend_on_when_the_note_was_struck() -> None:
    """Which is what makes them a figure about the stage rather than about the take.

    Two takes of one setting begin at whatever moment the recorder opened, and that
    rotates order `k` by `k` times one angle. A reading that did not take it out would
    return a different series every take and none of them would mean anything.
    """
    whole = a_curve(seconds=SECONDS + 1.0)
    early = whole[: int(SECONDS * SR)]
    # A shift that is not a whole number of carrier periods, so the carrier's own
    # angle really does move.
    late = whole[int(0.37 * SR) : int(0.37 * SR) + int(SECONDS * SR)]
    for one, other in zip(turns_of(early), turns_of(late), strict=True):
        assert min(abs(one - other), 360.0 - abs(one - other)) < 1.0


def test_an_order_carries_the_floor_it_is_standing_in(one_curve) -> None:
    """Read in its own take, so no other take has to be believed about the moment.

    The silence take is the other floor and it is a different claim: that one says
    what the chain returns with nothing played, this one says what this take holds
    between its own orders. Where the chain was quieter than the silence happened to
    be, this is the tighter of the two, and it is the one that needs no second take.
    """
    found = read(one_curve)
    first = next(row for row in found["readings"] if row["value"] == 0)
    beside = first["floor_between_orders_db"]
    assert beside is not None
    # The three orders this curve makes stand clear of it by a very long way, and the
    # orders above them are float residue and do not.
    assert min(first["orders_db"][k] - beside[k] for k in range(3)) > 100.0
    assert first["orders_db"][3] - beside[3] < 60.0
