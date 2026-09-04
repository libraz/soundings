"""Messages sent after the setting, which is what a switch has to be asked with.

The failure guarded here is an ordering one and it fails silently: the harness
sends its volume and its expression before it writes the setting, so a byte that
does nothing but decide whether a message is acted on has nothing left to act on.
Both settings sound alike and the null reads as a fact about the unit.

No hardware: what is under test is the bytes a move renders and the order a
stimulus carries them in.
"""

from __future__ import annotations

import pytest

from soundings import gestures, stimuli


def test_a_controller_move_is_one_message_on_the_stimulus_channel() -> None:
    assert gestures.cc(7, 40, "volume").messages(2) == [[0xB2, 7, 40]]


def test_a_bend_is_centred_on_the_middle_of_its_range() -> None:
    """Zero is no bend, not the bottom of the range: a move of 0 has to leave the
    note where it was, or every gesture carrying a bend is a gesture carrying a
    transposition nobody asked for."""
    assert gestures.Move("bend", (0,), "").messages(0) == [[0xE0, 0x00, 0x40]]
    assert gestures.Move("bend", (8191,), "").messages(0) == [[0xE0, 0x7F, 0x7F]]


def test_a_bend_past_the_end_of_the_range_is_held_at_the_end() -> None:
    """A wrapped bend is the worst kind of wrong value: it is legal MIDI, so
    nothing refuses it, and it moves the note the other way."""
    assert gestures.Move("bend", (99999,), "").messages(0) == [[0xE0, 0x7F, 0x7F]]


def test_a_bank_select_is_sent_with_the_program_change_that_commits_it() -> None:
    """This unit holds a bank select without changing anything until a program
    change arrives, and discards a pair it does not have. A bank select left
    without one sets a latch that throws away the next program change instead."""
    assert gestures.Move("bank", (8, 0, 12), "").messages(0) == [
        [0xB0, 0, 8],
        [0xB0, 32, 0],
        [0xC0, 12],
    ]


@pytest.mark.parametrize(
    ("kind", "select"),
    [("rpn", (101, 100)), ("nrpn", (99, 98))],
)
def test_a_parameter_move_parks_its_selector_afterwards(kind: str, select) -> None:
    """A selector left standing is how a data entry sent later by anything else
    lands at an address nobody named."""
    sent = gestures.Move(kind, (0x01, 0x21, 0x60), "").messages(0)

    assert sent[:3] == [[0xB0, select[0], 1], [0xB0, select[1], 0x21], [0xB0, 6, 0x60]]
    assert sent[3:] == [[0xB0, select[0], 0x7F], [0xB0, select[1], 0x7F]]


def test_an_unknown_kind_is_refused_rather_than_sent_as_nothing() -> None:
    """Returning no messages would leave the gesture silently short, and a switch
    asked under a gesture missing its message reads as a null."""
    with pytest.raises(KeyError):
        gestures.Move("aftertouch", (0,), "").messages(0)


def test_neither_gesture_stacks_two_attenuators() -> None:
    """Volume and expression cost about 20 dB each on this unit, and two of them
    together take the take down towards the noise the yardstick is measured
    from -- which turns a verdict into an inconclusive one."""
    for moves in (gestures.MOVED, gestures.RETUNED):
        quiet = [m for m in moves if m.kind == "cc" and m.data[0] in (7, 11) and m.data[1] < 127]
        assert len(quiet) <= 1


def test_the_switch_stimuli_are_the_plain_note_and_the_three_that_move() -> None:
    """The plain note answers a parameter that shapes the voice and cannot ask
    one that gates a message; the gestures ask the second and are a worse
    question for the first, since they move a great deal at once."""
    asked = stimuli.resolve(["switch"])

    assert [s.name for s in asked] == [
        "struck",
        "struck_moved",
        "struck_retuned",
        "struck_vibrato",
    ]
    assert asked[0].moves == ()
    assert all(s.moves for s in asked[1:])


def test_the_modulator_has_a_gesture_to_itself() -> None:
    """It is a free-running LFO, so it stops the takes repeating rather than
    changing what they hold -- measured, from 0.2 dB apart to 10.6. Anything
    sharing a gesture with it is judged against a yardstick nothing clears."""
    assert [m.data[0] for m in gestures.VIBRATO] == [1]
    for moves in (gestures.MOVED, gestures.RETUNED):
        assert not [m for m in moves if m.kind == "cc" and m.data[0] == 1]


def test_a_move_that_writes_the_address_under_test_is_taken_out() -> None:
    """Volume is stored at the part level, so a run asking the part level while
    the gesture sends volume puts both settings at the gesture's byte. Measured
    the first time these ran: the level written as 0 came back sounding 31.8 dB
    over the lead-in."""
    kept, dropped = gestures.without(gestures.MOVED, 0x19)

    assert [m.data[0] for m in dropped] == [7]
    assert 7 not in [m.data[0] for m in kept if m.kind == "cc"]


def test_a_bank_select_is_taken_out_for_either_byte_it_writes() -> None:
    """It commits a bank and a program together, which land at two addresses.
    Matching only one of them would leave the other measuring the gesture."""
    for offset in (0x00, 0x01):
        _, dropped = gestures.without(gestures.RETUNED, offset)
        assert [m.kind for m in dropped] == ["bank"]


def test_an_address_the_gesture_does_not_write_takes_nothing_out() -> None:
    """The gesture is the question, and dropping a message from it makes the
    question weaker. It must only happen where it has to."""
    kept, dropped = gestures.without(gestures.MOVED, 0x1D)

    assert kept == gestures.MOVED and dropped == []


def test_a_run_against_no_address_takes_nothing_out() -> None:
    """A controller run writes no address in the part block at all."""
    assert gestures.without(gestures.RETUNED, None) == (gestures.RETUNED, [])


def test_a_gesture_stimulus_says_what_it_moved_in_its_own_description() -> None:
    """A note played after six controllers have been moved is a different
    question from the same note under power-on defaults. A verdict whose
    description did not say so would read as the plain note's."""
    described = stimuli.CATALOGUE["struck_moved"].describe()

    assert "moved:" in described
    assert "volume" in described


def test_the_moves_travel_in_the_stimulus_json() -> None:
    """The record is the archive, and the state a verdict was taken in is part of
    the verdict."""
    written = stimuli.CATALOGUE["struck_retuned"].to_json()

    assert [m["kind"] for m in written["moves"]] == ["bank", "cc", "cc", "nrpn", "rpn", "bend"]


def test_the_plain_stimuli_move_nothing() -> None:
    """Adding the field must not have changed what any existing stimulus asks,
    or every verdict already in the archive was taken under a different question
    from the one the same name now names."""
    for name in ("struck", "released", "sustained", "unpitched", "wash", "deep", "struck_kit"):
        assert stimuli.CATALOGUE[name].moves == ()


def test_the_offset_is_read_from_the_address_against_the_stimulus_channel() -> None:
    """GS does not lay the parts out in channel order, so the block a channel's
    parameters live in is arithmetic rather than the channel number. An address
    matched against the wrong block would drop a move for a part nobody is
    listening to, and keep one for the part under test."""
    from argparse import Namespace

    from soundings.cli.sound import _part_offset

    args = Namespace(cc=None, address="40 11 19", channel=0)

    assert _part_offset(args, 0) == 0x19
    # Channel 2 is part 3, at 40 13, so the same address is another part's.
    assert _part_offset(args, 2) is None


def test_an_address_outside_the_part_block_takes_nothing_out() -> None:
    """An effect parameter or a drum setup byte is nothing a gesture writes."""
    from argparse import Namespace

    from soundings.cli.sound import _part_offset

    assert _part_offset(Namespace(cc=None, address="40 03 00", channel=0), 0) is None
    assert _part_offset(Namespace(cc=7, address=None, channel=0), 0) is None
