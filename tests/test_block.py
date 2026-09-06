"""Reading a block's per-address records into one answer about the block.

Two failures are guarded here and both read as a complete result. A directory
holding fewer records than the block has addresses looks exactly like a block
that answered; and under a gesture carrying a modulator, the channel that reads
one setting repeating worse than the other fires on the scatter of a free-running
phase, which looks exactly like a switch that gates it.

No hardware: what is under test is the reading of contrast records.
"""

from __future__ import annotations

import json

from soundings import block


def record(
    address: str,
    *,
    audible: bool = False,
    heard: tuple[str, ...] = (),
    deaf: tuple[str, ...] = ("struck",),
    inconclusive: tuple[str, ...] = (),
    unrepeatable: dict | None = None,
    values: tuple[int, int] = (0, 127),
    grounds: dict | None = None,
) -> dict:
    """A contrast record as the file holds one.

    `grounds` says which of the three channels carried each stimulus, defaulting
    to the repeatability channel for anything heard. That default is the case the
    guard is about, and it is what the hardware produced for every verdict the
    guard had to be widened to catch.
    """
    names = list(heard) + list(deaf) + list(inconclusive)
    grounds = grounds or {}
    return {
        "address": address,
        "values": list(values),
        "audible": audible,
        "heard_by": list(heard),
        "not_heard_by": list(deaf),
        "inconclusive_under": list(inconclusive),
        "by_stimulus": [
            {
                "stimulus_name": name,
                "each_setting_unrepeatable_db": (unrepeatable or {}).get(name, [-50.0, -50.0]),
                "changed_the_shape": "shape" in grounds.get(name, []),
                "changed_the_level": "level" in grounds.get(name, []),
                "changed_the_repeatability": (
                    "repeatability" in grounds.get(name, ["repeatability"])
                    if name in heard
                    else False
                ),
            }
            for name in names
        ],
    }


def written(tmp_path, name: str, body: dict):
    (tmp_path / f"{name}.json").write_text(json.dumps(body))


def test_an_address_the_plain_note_heard_is_answered_by_the_plain_note() -> None:
    """A gesture asks in a state the verdict then holds only in, so it is not a
    peer of the plain note -- it is what is reached for when the plain note could
    not hear anything."""
    plain = {"40 11 08": block.read_one(record("40 11 08", audible=True, heard=("struck",)))}
    gesture = {"40 11 08": block.read_one(record("40 11 08", deaf=("struck_moved",)))}

    joined = block.join(plain, gesture)

    assert [f.heard_by for f in joined] == [["struck"]]


def test_an_address_the_plain_note_missed_is_answered_by_the_gesture() -> None:
    """Which is the whole reason the second pass exists: a byte that only decides
    whether a message is received has nothing to receive under a plain note."""
    plain = {"40 11 0C": block.read_one(record("40 11 0C"))}
    gesture = {
        "40 11 0C": block.read_one(
            record("40 11 0C", audible=True, heard=("struck_moved",), deaf=())
        )
    }

    joined = block.join(plain, gesture)

    assert joined[0].audible and joined[0].heard_by == ["struck_moved"]
    # What the plain note tried is kept: a verdict reached only under a gesture
    # is a narrower claim, and the list of what did not hear it says so.
    assert "struck" in joined[0].not_heard_by


def test_the_modulator_gesture_is_not_read_as_a_gate_on_scatter_alone() -> None:
    """Both settings carry the modulator, so the channel has nothing to detect
    and the gap between them is a free-running phase. Measured on the first
    address asked: 29.2 dB against 14.5 -- a wide gap, on an address the plain
    note found no difference in at all."""
    plain = {
        "40 11 00": block.read_one(record("40 11 00", unrepeatable={"struck": [-57.9, -61.43]}))
    }
    gesture = {
        "40 11 00": block.read_one(
            record(
                "40 11 00",
                audible=True,
                heard=("struck_vibrato",),
                deaf=(),
                unrepeatable={"struck_vibrato": [-29.19, -14.5]},
            )
        )
    }

    joined = block.join(plain, gesture)

    assert not joined[0].audible
    assert joined[0].discounted == ["struck_vibrato"]
    assert "struck_vibrato" in joined[0].inconclusive_under


def test_a_setting_that_really_is_steady_is_read_as_a_gate() -> None:
    """The guard has to have an outside, or the modulation gesture can never
    answer anything and the switch it exists for is unaskable."""
    plain = {
        "40 11 0B": block.read_one(record("40 11 0B", unrepeatable={"struck": [-57.9, -61.4]}))
    }
    gesture = {
        "40 11 0B": block.read_one(
            record(
                "40 11 0B",
                audible=True,
                heard=("struck_vibrato",),
                deaf=(),
                unrepeatable={"struck_vibrato": [-55.0, -8.0]},
            )
        )
    }

    joined = block.join(plain, gesture)

    assert joined[0].audible and joined[0].heard_by == ["struck_vibrato"]
    assert joined[0].discounted == []


def test_the_guard_needs_the_plain_pass_and_refuses_without_it() -> None:
    """The plain figure is what "steady" is measured against. With none there is
    nothing to compare, and reading the gap as a gate would be the claim the
    guard exists to stop being made unexamined."""
    gesture = block.read_one(
        record(
            "40 11 00",
            audible=True,
            heard=("struck_vibrato",),
            deaf=(),
            unrepeatable={"struck_vibrato": [-55.0, -8.0]},
        )
    )

    assert block.scatter_not_a_gate("struck_vibrato", gesture, None)


def test_a_verdict_backed_by_the_sound_itself_is_left_alone() -> None:
    """The guard is about the repeatability channel and only about it. A change
    of shape or of level is a measurement of the sound, so it stands however the
    takes of either setting happened to scatter.

    Written first as "the other gestures carry no modulator, so their
    repeatability channel means what it always meant". The hardware refused that:
    across three part blocks holding the same parameters, struck_again and
    struck_retuned produced sixteen single-witness verdicts on a different set of
    addresses in each, every one of them on the repeatability channel alone."""
    heard_by_shape = block.read_one(
        record(
            "40 11 0C",
            audible=True,
            heard=("struck_moved",),
            deaf=(),
            unrepeatable={"struck_moved": [-3.0, -30.0]},
            grounds={"struck_moved": ["shape"]},
        )
    )

    assert not block.scatter_not_a_gate("struck_moved", heard_by_shape, None)


def test_a_second_stimulus_scattering_is_caught_the_same_way() -> None:
    """Measured: struck_again on 40 12 0E, settings scattering -1.68 and -13.85
    dB against a plain note's -56 dB on this unit, called audible."""
    plain = {
        "40 12 0E": block.read_one(record("40 12 0E", unrepeatable={"struck": [-57.9, -61.4]}))
    }
    poly = {
        "40 12 0E": block.read_one(
            record(
                "40 12 0E",
                audible=True,
                heard=("struck_again",),
                deaf=("struck_pair",),
                unrepeatable={"struck_again": [-1.68, -13.85]},
            )
        )
    }

    (joined,) = block.join(plain, {}, poly)

    assert not joined.audible
    assert joined.discounted == ["struck_again"]
    assert "struck_again" in joined.inconclusive_under


def test_a_block_shorter_than_its_plan_says_so() -> None:
    """A directory of records answers about the records. Forty-five records of a
    forty-seven address block reads exactly like a complete answer, so the count
    is against the plan rather than against the files present."""
    planned = {
        "ask": [{"address": "40 11 00"}, {"address": "40 11 01"}, {"address": "40 11 02"}],
        "cannot_be_asked": [{"address": "40 11 17", "why": "one value"}],
    }
    found = [block.read_one(record("40 11 00")), block.read_one(record("40 11 02"))]

    coverage = block.against_plan(found, planned)

    assert coverage == {
        "planned": 3,
        "answered": 2,
        "never_asked": ["40 11 01"],
        "cannot_be_asked": planned["cannot_be_asked"],
    }


def test_a_directory_is_read_by_the_address_each_record_holds(tmp_path) -> None:
    """Not by the filename: the record carries the address it was taken at, and a
    file named for another is a record attributed to an address it is not of."""
    written(tmp_path, "second", record("40 11 02", audible=True, heard=("struck",)))
    written(tmp_path, "first", record("40 11 00"))

    found = block.survey(tmp_path)

    assert sorted(found) == ["40 11 00", "40 11 02"]
    assert found["40 11 02"].audible


def test_what_is_left_to_try_is_named(tmp_path) -> None:
    """It is the next run's list, and a count is not a list."""
    plain = {
        "40 11 00": block.read_one(record("40 11 00")),
        "40 11 02": block.read_one(record("40 11 02", audible=True, heard=("struck",))),
    }

    joined = block.join(plain, {})

    assert [f.address for f in joined if f.still_open] == ["40 11 00"]


def test_an_address_nothing_could_measure_is_not_a_null(tmp_path) -> None:
    """A yardstick nothing could clear says nothing about the parameter, and
    filing it with the nulls would state a fact on a measurement with no power to
    find one."""
    plain = {"40 11 00": block.read_one(record("40 11 00", deaf=(), inconclusive=("struck",)))}

    joined = block.join(plain, {})

    assert joined[0].verdict == "could not be measured"
    assert joined[0].still_open


def test_a_balance_that_moved_counts_as_the_parameter_having_reached_the_path() -> None:
    """It is invisible to a comparison made in one channel, and does not read as
    nothing there: a balance landing somewhere new on each take reads as the unit
    failing to repeat. Measured on the part panpot, whose plain verdict was that
    nothing could be measured."""
    found = [block.read_one(record("40 11 1C", deaf=(), inconclusive=("struck",)))]
    measured = {
        "runs": [
            {
                "name": "40-11-1C",
                "stimulus_name": "struck",
                "moved_between_settings": True,
                "did_not_repeat_within_a_setting": True,
            }
        ]
    }

    (joined,) = block.with_balance(found, measured)

    assert joined.audible and joined.heard_by == ["struck (balance)"]
    assert joined.inconclusive_under == []


def test_a_balance_that_held_still_changes_no_verdict() -> None:
    """Forty-three of forty-six addresses showed a spread of 0.0 dB, so the route
    has to leave a null a null."""
    found = [block.read_one(record("40 11 30"))]
    measured = {
        "runs": [
            {
                "name": "40-11-30",
                "stimulus_name": "struck",
                "moved_between_settings": False,
                "did_not_repeat_within_a_setting": False,
            }
        ]
    }

    (joined,) = block.with_balance(found, measured)

    assert not joined.audible and joined.heard_by == []


def test_a_move_the_gesture_could_not_carry_travels_with_the_verdict() -> None:
    """A gesture cannot send a message that writes the address under test, so the
    gesture asked there was one message short and its null is narrower. For an
    address whose only route to being heard is the message that stores into it,
    that is narrower to the point of being unanswerable this way."""
    body = record("40 11 00", deaf=("struck_moved", "struck_retuned"))
    body["left_out_of_the_gesture"] = {
        "struck_retuned": [{"kind": "bank", "data": [8, 0, 12], "stores_at_part_offset": [0, 1]}]
    }

    written = block.read_one(body).to_json()

    assert written["left_out_of_the_gesture"]["struck_retuned"][0]["kind"] == "bank"
    assert block.WHY_LEFT_OUT in written["why_left_out"]


def test_what_each_pass_left_out_is_kept_when_they_are_joined() -> None:
    """The caveat belongs to the address, and either pass may have carried one."""
    first = record("40 11 1C", deaf=("struck",))
    second = record("40 11 1C", deaf=("struck_moved",))
    second["left_out_of_the_gesture"] = {"struck_moved": [{"kind": "cc", "data": [10, 0]}]}

    (joined,) = block.join(
        {"40 11 1C": block.read_one(first)}, {"40 11 1C": block.read_one(second)}
    )

    assert "struck_moved" in joined.left_out


def test_an_address_only_two_notes_could_hear_is_answered_by_the_polyphony_pass() -> None:
    """Whether a part sounds two voices at once is inaudible under one note however
    the address is set, so the one-note passes' null is a fact about the stimulus."""
    plain = {"40 11 14": block.read_one(record("40 11 14", deaf=("struck",)))}
    gesture = {"40 11 14": block.read_one(record("40 11 14", deaf=("struck_moved",)))}
    poly = {
        "40 11 14": block.read_one(
            record("40 11 14", audible=True, heard=("struck_pair",), deaf=("struck_again",))
        )
    }

    (joined,) = block.join(plain, gesture, poly)

    assert joined.audible and joined.verdict == "audible"
    assert joined.heard_by == ["struck_pair"]
    # Every stimulus that failed to hear it stays on the record: the verdict is
    # what one of them heard, not a claim that the others were not asked.
    assert set(joined.not_heard_by) == {"struck", "struck_moved", "struck_again"}


def test_the_two_rescues_are_peers_and_both_are_named() -> None:
    """Neither rescue outranks the other, so an address both reached carries both."""
    plain = {"40 11 09": block.read_one(record("40 11 09", deaf=("struck",)))}
    gesture = {
        "40 11 09": block.read_one(
            record("40 11 09", audible=True, heard=("struck_moved",), deaf=())
        )
    }
    poly = {
        "40 11 09": block.read_one(
            record("40 11 09", audible=True, heard=("struck_again",), deaf=())
        )
    }

    (joined,) = block.join(plain, gesture, poly)

    assert joined.heard_by == ["struck_moved", "struck_again"]


def test_a_polyphony_verdict_whose_steady_setting_is_steady_survives() -> None:
    """The guard has to have an outside here too, or the only stimulus that can
    ask about polyphony can never answer and the question is unaskable.

    The figures are the measured ones for 40 11 24, the address struck_again is
    the sole witness for: one setting at -66.34 dB is as steady as a plain note
    on this unit, and the other falls apart at -11.35. That asymmetry is what a
    gate looks like, and it is the shape none of the sixteen discarded verdicts
    had -- in those, both settings scattered."""
    plain = {
        "40 11 24": block.read_one(record("40 11 24", unrepeatable={"struck": [-56.4, -58.1]}))
    }
    poly = {
        "40 11 24": block.read_one(
            record(
                "40 11 24",
                audible=True,
                heard=("struck_again",),
                deaf=("struck_pair",),
                unrepeatable={"struck_again": [-66.34, -11.35]},
            )
        )
    }

    (joined,) = block.join(plain, {}, poly)

    assert joined.audible and joined.heard_by == ["struck_again"]
    assert joined.discounted == []


def test_a_block_read_without_a_polyphony_pass_is_unchanged() -> None:
    """The third pass is optional, and a block asked before it existed reads the
    same way afterwards -- otherwise every earlier record would silently move."""
    plain = {"40 11 08": block.read_one(record("40 11 08", audible=True, heard=("struck",)))}
    gesture = {"40 11 08": block.read_one(record("40 11 08", deaf=("struck_moved",)))}

    assert [f.to_json() for f in block.join(plain, gesture)] == [
        f.to_json() for f in block.join(plain, gesture, {})
    ]


def test_a_control_kept_beside_the_verdicts_is_not_one_of_them() -> None:
    """A run keeps its control in the same directory, and the control is another block's
    address. Counting it would make the block a byte wider than it is."""
    planned = {
        "ask": [{"address": "40 03 17"}],
        "cannot_be_asked": [{"address": "40 03 1A", "why": "accepts one value"}],
    }
    found = {
        "40 03 17": object(),
        "40 03 1A": object(),
        "40 42 22": object(),
    }
    kept, outside = block.split_by_plan(found, planned)
    assert sorted(kept) == ["40 03 17", "40 03 1A"]
    assert outside == ["40 42 22"]


def test_a_plan_naming_its_unaskable_address_as_a_string_still_matches() -> None:
    """Two stages write that list differently, and neither shape may lose a record."""
    planned = {"ask": [], "cannot_be_asked": ["40 03 1A (accepts one value)"]}
    kept, outside = block.split_by_plan({"40 03 1A": object()}, planned)
    assert sorted(kept) == ["40 03 1A"]
    assert outside == []


def test_an_address_asked_over_a_guessed_span_says_so_in_the_block(tmp_path):
    """The weakest null in a block, and it arrived looking like every other one.

    The write probe never writes an address whose original it could not read,
    because there would be nothing to put back, so the plan asks it over the
    whole seven-bit span and carries a caveat saying as much. That caveat lived
    in the plan, which is what a run was aimed by, and not in the block record,
    which is what anyone reads afterwards. Seven addresses of the system effect
    block are in exactly that position."""
    verdict = block.AddressVerdict(address="40 01 41", values=[0, 127], audible=False)
    planned = {
        "ask": [
            {"address": "40 01 41", "caveat": "asked over the whole seven-bit span"},
            {"address": "40 01 30"},
        ]
    }

    (kept,) = block.with_the_plans_caveats([verdict], planned)

    assert kept.to_json()["how_it_had_to_be_asked"] == "asked over the whole seven-bit span"


def test_an_address_the_plan_had_nothing_to_say_about_carries_nothing(tmp_path):
    """The key is absent rather than present and empty, so a record that says
    something about how an address was asked is one where something was said."""
    verdict = block.AddressVerdict(address="40 01 30", values=[0, 7], audible=True)

    (kept,) = block.with_the_plans_caveats([verdict], {"ask": [{"address": "40 01 30"}]})

    assert "how_it_had_to_be_asked" not in kept.to_json()


def test_the_method_describes_a_gesture_pass_only_where_one_was_folded_in() -> None:
    """The gesture sentence was part of one fixed string, so a block folded from a
    single pass published a method describing a second pass that never ran.

    Nothing downstream compares a method against the directories the fold was
    handed, so the sentence is the record's own account of itself and there was
    nothing to catch it -- which is the whole reason a record may make claims.
    """
    alone = block.method_for(gesture=False)
    both = block.method_for(gesture=True)

    gesture_sentence = "was asked again with messages moved after the setting"
    assert gesture_sentence not in alone
    assert gesture_sentence in both
    # The rest of the account is the same either way, and each form is one whole
    # literal rather than pieces joined at run time: the archive gate pins a
    # published method to the source verbatim, so an assembled sentence is
    # pinned to nothing.
    for sentence in alone.split(". "):
        assert sentence in both


def test_a_pass_carries_the_state_its_verdicts_were_taken_in(tmp_path):
    """A system effect is reached only through a part's send to it, so the send a
    pass raised first is part of every verdict it produced.

    Read back from the records rather than written beside them: a driver says
    what it meant to raise and the records say what was raised, and the two have
    already parted company here -- one pass raised part 1's send while its
    stimulus sounded on part 10.
    """
    for name, prepared in (("a", "7F"), ("b", "7F"), ("c", "40")):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "prepared": [{"address": "40 11 21", "bytes": prepared}],
                    "by_stimulus": [{"stimulus_name": "deep"}],
                }
            )
        )

    found = block.states(tmp_path)

    assert found == [
        {"prepared": ["40 11 21 = 7F"], "stimuli": ["deep"], "addresses": 2},
        {"prepared": ["40 11 21 = 40"], "stimuli": ["deep"], "addresses": 1},
    ]


def test_a_pass_that_prepared_nothing_says_so_rather_than_being_left_out(tmp_path):
    """The state the unit powers up in is a state, and a block asked in it is
    making a claim bounded by it exactly as one asked with a send raised is."""
    (tmp_path / "a.json").write_text(json.dumps({"by_stimulus": [{"stimulus_name": "struck"}]}))

    assert block.states(tmp_path) == [{"prepared": [], "stimuli": ["struck"], "addresses": 1}]
