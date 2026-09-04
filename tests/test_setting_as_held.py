"""What the two settings are is what the unit holds, not what the run asked for.

The guard this replaces refused any address whose readback differed from the byte
written, and on hardware that took seven addresses out of a twelve address pass:
one clamped 127 to 95, one clamped 2 to 1, one answered 127 with 2. Every one of
those is a pair that lands somewhere different and is therefore measurable, and
the plan's range is itself derived from a probe rather than given, so a byte
coming back changed says the plan guessed narrow rather than that the address
cannot be asked.

What is still refused is a pair that collapses to one byte, which is the failure
the readback exists for: two sets of takes of the same setting, reading as a
parameter that does nothing.
"""

from __future__ import annotations

from soundings.cli import session


class FakeLink:
    """A unit that clamps writes to a ceiling and answers one-byte reads."""

    def __init__(self, ceiling: int = 0x7F, *, answers: bool = True) -> None:
        self.ceiling = ceiling
        self.answers = answers
        self.held = 0
        self.sent: list[list[int]] = []

    def send(self, message) -> None:
        self.sent.append(list(message))

    def receive(self, timeout: float = 0.0) -> list[int]:
        # A list either way: the link returns what arrived, and nothing arriving
        # is an empty one rather than a None. A fake that returns None instead
        # tests a contract the real link does not have.
        return []

    def exchange(self, message) -> list[int]:
        from soundings import roland

        if not self.answers:
            return []
        return roland.dt1("40 11 18", [self.held], device_id=0x10)

    def write(self, address: str, value: int) -> None:
        self.held = min(value, self.ceiling)


def test_a_clamped_write_reports_the_byte_the_unit_kept() -> None:
    link = FakeLink(ceiling=0x5F)
    link.write("40 11 18", 127)

    assert session.holds(link, "40 11 18", device_id=0x10) == 0x5F


def test_an_address_that_answers_nothing_is_none_rather_than_a_refusal() -> None:
    """Two addresses in a part block are readable only as part of a wider region."""
    link = FakeLink(answers=False)

    assert session.holds(link, "40 11 18", device_id=0x10) is None


def test_a_pair_that_clamps_apart_is_still_two_settings() -> None:
    """The measurement the old guard threw away: 0 and 95 are not the same byte."""
    link = FakeLink(ceiling=0x5F)

    landed = []
    for value in (0, 127):
        link.write("40 11 18", value)
        landed.append(session.holds(link, "40 11 18", device_id=0x10))

    assert landed == [0x00, 0x5F]
    assert landed[0] != landed[1]


def test_a_pair_that_clamps_together_is_one_setting() -> None:
    """The failure the readback is for, and the only one left that refuses a run."""
    link = FakeLink(ceiling=0x01)

    landed = []
    for value in (2, 127):
        link.write("40 11 18", value)
        landed.append(session.holds(link, "40 11 18", device_id=0x10))

    assert landed[0] == landed[1]


class WrongAddressLink(FakeLink):
    """A unit whose reply is for a different address than the one asked about.

    Not hypothetical: a one-byte read of an address readable only inside a wider
    region came back with a neighbour's byte, and the run refused a measurable
    address on the strength of it.
    """

    def exchange(self, message) -> list[int]:
        from soundings import roland

        return roland.dt1("40 11 19", [0x5F], device_id=0x10)


def test_a_reply_for_another_address_is_not_an_answer() -> None:
    assert session.holds(WrongAddressLink(), "40 11 18", device_id=0x10) is None


def test_a_reply_for_the_address_asked_about_still_answers() -> None:
    """The check must not swallow the readings it is protecting."""
    link = FakeLink()
    link.write("40 11 18", 0x21)

    assert session.holds(link, "40 11 18", device_id=0x10) == 0x21
