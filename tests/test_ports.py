"""Crediting a byte to the message that wrote it, across a cable move.

The sending phase cannot read anything back, so nothing watches a byte move at
the time. The whole attribution rests on each stimulus carrying a value no other
stimulus carried, which makes the uniqueness check part of the measurement rather
than a tidiness rule -- two stimuli at one value would produce a table that looks
exactly like a correct one and names the wrong message.

No hardware.
"""

from __future__ import annotations

import pytest

from soundings import ports
from soundings.writeback import NEVER_WRITE


def test_no_two_stimuli_in_the_catalogue_carry_the_same_value():
    """The property everything downstream is read through. It is asserted against
    the real catalogue rather than a constructed one, because the catalogue is
    what gets sent."""
    assert ports.repeated_values(ports.catalogue()) == {}


def test_a_repeated_value_names_both_stimuli_rather_than_keeping_one():
    """A run that silently kept the last of them would credit its addresses to
    one message and report the other as landing nowhere -- a false positive and a
    false negative from the same collision."""
    doubled = [
        ports.Sent(label="one", kind="cc", value=0x11),
        ports.Sent(label="two", kind="cc", value=0x11),
        ports.Sent(label="three", kind="cc", value=0x12),
    ]
    assert ports.repeated_values(doubled) == {0x11: ["one", "two"]}


def test_the_catalogue_writes_to_nothing_on_the_never_write_list():
    addresses = {address for address, _ in ports.WRITE_TARGETS}
    assert not addresses & NEVER_WRITE


def test_the_per_channel_sweep_is_sent_before_the_extras_on_the_same_channel():
    """The extras share the first channel with the sweep, so whichever goes last
    is what the byte holds. Sent the other way round, the sweep would overwrite an
    extra's value on its own address and the extra would read as landing
    nowhere."""
    labels = [s.label for s in ports.catalogue(channels=2)]
    assert labels.index("CC7 on channel 1") < labels.index("CC10 on channel 1")


def test_a_moved_byte_holding_a_value_the_run_sent_is_credited_to_that_stimulus():
    sent = [ports.Sent(label="CC7 on channel 1", kind="cc", value=0x11)]
    landed, unexplained, unknown = ports.locate(sent, {"50 11 19": 0x64}, {"50 11 19": 0x11})
    assert landed == {"CC7 on channel 1": ["50 11 19"]}
    assert unexplained == {} and unknown == []


def test_a_byte_that_already_held_the_value_is_credited_to_nothing():
    """It did not move, so nothing shows this run put it there. Crediting it
    would let a stimulus be attributed to an address it never reached, and the
    coincidence is not rare: three quarters of this unit's power-on bytes hold one
    of four values."""
    sent = [ports.Sent(label="CC7 on channel 1", kind="cc", value=0x11)]
    landed, unexplained, _ = ports.locate(sent, {"40 11 19": 0x11}, {"40 11 19": 0x11})
    assert landed == {"CC7 on channel 1": []}
    assert unexplained == {}


def test_a_byte_that_moved_to_a_value_nobody_sent_is_kept_rather_than_dropped():
    """A program change loads a whole part, so bytes carrying no value the run
    sent are expected. Discarding them would leave a run unable to tell a
    knock-on effect from a byte it never noticed moving."""
    sent = [ports.Sent(label="program change on channel 1", kind="channel", value=0x27)]
    landed, unexplained, _ = ports.locate(
        sent, {"40 11 01": 0x00, "40 12 02": 0x00}, {"40 11 01": 0x27, "40 12 02": 0x51}
    )
    assert landed == {"program change on channel 1": ["40 11 01"]}
    assert unexplained == {"40 12 02": [0x00, 0x51]}


def test_a_byte_with_no_baseline_is_neither_credited_nor_called_unexplained():
    """It has not been shown to have moved. Counting it as a change would report
    a gap in an earlier capture as something this run did."""
    sent = [ports.Sent(label="CC7 on channel 1", kind="cc", value=0x11)]
    landed, unexplained, unknown = ports.locate(
        sent, {"40 00 00": 0x00}, {"40 00 00": 0x00, "51 20 00": 0x11}
    )
    assert landed == {"CC7 on channel 1": []}
    assert unexplained == {}
    assert unknown == ["51 20 00"]


def test_a_snapshot_sharing_no_address_with_the_baseline_is_refused():
    """The failure this guards is not an error but a plausible answer. Every read
    failing produces no landings and no moved bytes, which is exactly what an
    input that stores nothing produces, and the record cannot tell them apart. It
    was produced once, by a link whose replies had fallen a message behind: all
    854 regions were dropped and the comparison announced no differences."""
    sent = [ports.Sent(label="CC7 on channel 1", kind="cc", value=0x11)]
    with pytest.raises(ports.NothingWasRead):
        ports.locate(sent, {"40 11 19": 0x64}, {})


def test_a_snapshot_that_read_one_byte_the_baseline_holds_is_compared():
    """The guard must not be able to refuse a run that read anything at all, or a
    genuine finding on a short read is thrown away as a fault."""
    sent = [ports.Sent(label="CC7 on channel 1", kind="cc", value=0x11)]
    landed, _, _ = ports.locate(sent, {"40 11 19": 0x64}, {"40 11 19": 0x11})
    assert landed == {"CC7 on channel 1": ["40 11 19"]}


def test_the_blocks_reached_are_counted_by_top_byte():
    """The verdict of the whole run: which block a set of channel messages
    reaches is what the input under test addresses."""
    landed = {"a": ["50 11 19", "50 12 19"], "b": ["40 11 1A"], "c": []}
    assert ports.blocks_reached(landed) == {"40": 1, "50": 2}


def test_a_run_that_landed_nothing_anywhere_reaches_no_blocks():
    """An input that stores nothing is a possible outcome and must not come back
    looking like an empty dictionary of something else."""
    assert ports.blocks_reached({"a": [], "b": []}) == {}


def test_the_summary_names_the_stimuli_that_landed_nowhere():
    landed = {"CC7 on channel 1": ["50 11 19"], "CC11 on channel 1": []}
    text = ports.summarise(landed, {}, [])
    assert "50 11 19" in text
    assert "1 stimuli landed nowhere: CC11 on channel 1" in text
    assert "50 (1 bytes)" in text
