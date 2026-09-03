"""The insertion-effect survey, against a machine that does what the test says.

The survey rests on one behaviour of the unit -- a type it has stores verbatim,
a type it does not leaves something else -- and on one property of the address:
it takes two bytes at once and ignores a single one. Both are measured on the
SC-8850 and neither is checkable here; what is checkable is that the survey reads
those answers the way it claims to, including the ones it is supposed to refuse
to interpret.

No hardware. The fake stands in for the link, not for any particular unit.
"""

from __future__ import annotations

from soundings import efxmap, roland

TYPE = (0x40, 0x03, 0x00)
PARAMETERS = (0x40, 0x03, 0x03)
SENDS = (0x40, 0x03, 0x17)


class FakeUnit:
    """A machine holding a stated list of effects, each with its own settings.

    Selecting an effect it has loads that effect's parameters, which is the
    behaviour the map is built on. Selecting one it does not have leaves
    `fallback` in the type -- whatever the test says that is.
    """

    def __init__(self, effects: dict[tuple[int, int], int], *, fallback=(0, 0), deaf=()):
        self.effects = effects
        self.fallback = list(fallback)
        self.deaf = set(deaf)
        self.type = [0, 0]
        self.parameters = [0] * efxmap.PARAMETER_COUNT
        self.sends = [0, 0, 0]
        self.writes: list[list[int]] = []

    def send(self, message: list[int]) -> None:
        raw = list(message)
        if len(raw) < 9 or raw[4] != roland.CMD_DT1:
            return
        address, data = (raw[5], raw[6], raw[7]), raw[8:-2]
        if address != TYPE:
            return
        self.writes.append(list(data))
        # One byte is not a type. The address takes the pair or nothing.
        if len(data) != 2:
            return
        pair = (data[0], data[1])
        if pair in self.effects:
            self.type = list(pair)
            seed = self.effects[pair]
            self.parameters = [(seed + i) % 128 for i in range(efxmap.PARAMETER_COUNT)]
            self.sends = [seed % 128, (seed + 1) % 128, (seed + 2) % 128]
        else:
            self.type = list(self.fallback)

    def exchange(self, request: list[int], timeout: float = 0.0) -> list[int]:
        if len(request) < 11 or request[4] != roland.CMD_RQ1:
            return []
        address = (request[5], request[6], request[7])
        size = (request[8] << 14) | (request[9] << 7) | request[10]
        if address == TYPE and tuple(self.type) in self.deaf:
            return []
        data = {TYPE: self.type, PARAMETERS: self.parameters, SENDS: self.sends}.get(address)
        return [] if data is None else roland.dt1(address, data[:size])

    def receive(self, timeout: float = 0.0) -> list[int]:
        return []


def survey_of(unit: FakeUnit, highs) -> efxmap.Survey:
    asker = efxmap.Asker(unit, device_id=0x10, settle=0.0)
    return efxmap.survey(asker, high_bytes=highs)


def test_an_effect_the_unit_has_is_accepted_with_its_settings() -> None:
    found = survey_of(FakeUnit({(2, 1): 40}), [2])
    assert [e.number for e in found.accepted] == ["02 01"]
    effect = found.accepted[0]
    assert effect.parameters == [(40 + i) % 128 for i in range(efxmap.PARAMETER_COUNT)]
    assert effect.sends == [40, 41, 42]


def test_selecting_a_different_effect_gives_different_settings() -> None:
    """The settings have to come from the type just selected, not from whatever
    the previous one left, or every effect in the map reads the same."""
    found = survey_of(FakeUnit({(2, 0): 10, (2, 1): 90}), [2])
    first, second = found.accepted
    assert first.parameters != second.parameters


def test_a_refusal_is_recorded_as_what_it_left_rather_than_as_a_bare_count() -> None:
    """A fallback that varied by family would be invisible in a count."""
    found = survey_of(FakeUnit({(2, 0): 10}, fallback=(0x7F, 0x7F)), [2])
    assert found.refusals == {"7F 7F": 127}
    assert len(found.accepted) == 1


def test_a_family_whose_lowest_member_is_not_zero_is_still_found() -> None:
    """Why the sweep asks every low byte. Sampling a family at its zero member
    would report this one as absent, and absence is what the map asserts."""
    found = survey_of(FakeUnit({(5, 9): 3, (5, 11): 4}), [5])
    assert [e.number for e in found.accepted] == ["05 09", "05 0B"]


def test_the_type_is_written_as_a_pair_and_never_as_one_byte() -> None:
    """A single byte does nothing on the real address, so a survey that wrote one
    would report the whole space as refusing."""
    unit = FakeUnit({(2, 0): 10})
    survey_of(unit, [2])
    assert unit.writes
    assert all(len(w) == 2 for w in unit.writes)


def test_an_unanswered_read_is_not_counted_as_a_refusal() -> None:
    """A dropped reply and a rejected type are different facts and the second is
    a claim about the unit; merging them would put silence in the map as data."""
    found = survey_of(FakeUnit({(2, 0): 10}, deaf=[(0, 0)]), [2])
    assert found.unread and all(n.startswith("02 ") for n in found.unread)
    assert found.refusals == {}
    assert [e.number for e in found.accepted] == ["02 00"]


def test_a_narrowed_sweep_says_it_was_narrowed() -> None:
    found = survey_of(FakeUnit({(2, 0): 10}), [2])
    assert not found.exhaustive
    assert "means nothing" in found.to_json()["coverage"]
    assert "not asked for" in efxmap.summarise(found)


def test_a_whole_sweep_says_it_was_whole() -> None:
    found = survey_of(FakeUnit({(2, 0): 10}), range(128))
    assert found.exhaustive
    assert found.asked == 16384
    assert found.to_json()["coverage"] == "Every type number was asked."
