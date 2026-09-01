"""Where a channel's settings live, and putting back what a scan wrote."""

from __future__ import annotations

import pytest

from soundings import parts, roland


def test_channel_10_comes_first():
    """GS puts the drum part at index 0, which is the whole reason for the arithmetic."""
    assert parts.part_block(parts.TONE, 9) == (0x40, 0x10, 0x00)


@pytest.mark.parametrize(
    ("channel", "index"),
    [(0, 1), (1, 2), (8, 9), (9, 0), (10, 10), (15, 15)],
)
def test_every_channel_lands_on_its_own_part(channel: int, index: int):
    assert parts.part_block(parts.TONE, channel) == (0x40, 0x10 | index, 0x00)
    assert parts.part_block(parts.MAPPED, channel) == (0x40, 0x40 | index, 0x00)


def test_the_sixteen_channels_occupy_sixteen_distinct_blocks():
    blocks = {parts.part_block(parts.TONE, c) for c in range(16)}
    assert len(blocks) == 16


class FakeLink:
    """A link that answers RQ1 out of a dict, and records what was sent to it."""

    def __init__(self, memory: dict[tuple[int, int, int], list[int]]):
        self.memory = memory
        self.sent: list[list[int]] = []

    def receive(self, timeout=0.0):
        return []

    def send(self, message):
        self.sent.append(list(message))

    def exchange(self, request, timeout=1.0):
        address = (request[5], request[6], request[7])
        size = request[10]
        data = self.memory.get(address)
        if data is None:
            return []
        return roland.dt1(address, data[:size], device_id=request[2])


def test_remember_drops_addresses_that_will_not_answer():
    link = FakeLink({(0x40, 0x11, 0x30): [0x40]})
    kept = parts.Restorer(link, device_id=0x10).remember([(0x40, 0x11, 0x30), (0x40, 0x11, 0x31)])
    assert kept == {(0x40, 0x11, 0x30): 0x40}


def test_put_back_reports_a_byte_it_could_not_restore():
    """The unit keeps its own value, so the re-read disagrees and must say so."""
    link = FakeLink({(0x40, 0x11, 0x30): [0x7F]})
    said = parts.Restorer(link, device_id=0x10, settle=0.0).put_back({(0x40, 0x11, 0x30): 0x40})
    assert said.startswith("NOT restored")
    assert "40 11 30" in said


def test_put_back_confirms_only_after_re_reading():
    memory = {(0x40, 0x11, 0x30): [0x00]}
    link = FakeLink(memory)

    def send(message):
        link.sent.append(list(message))
        if len(message) > 8 and message[4] == roland.CMD_DT1:
            memory[(message[5], message[6], message[7])] = list(message[8:-2])

    link.send = send
    said = parts.Restorer(link, device_id=0x10, settle=0.0).put_back({(0x40, 0x11, 0x30): 0x40})
    assert said == "1 bytes put back and re-read"
    assert memory[(0x40, 0x11, 0x30)] == [0x40]


def test_clear_bank_latch_says_so_when_the_part_cannot_be_read():
    said = parts.Restorer(FakeLink({}), device_id=0x10).clear_bank_latch(0)
    assert "left as the scan left it" in said


def test_clear_bank_latch_commits_the_tone_the_part_already_holds():
    link = FakeLink({(0x40, 0x11, 0x00): [0x08, 0x50], (0x40, 0x41, 0x00): [0x02]})
    said = parts.Restorer(link, device_id=0x10).clear_bank_latch(0)
    assert said == "committed bank 8, map 2, program 80"
    # Bank MSB, then the CC32 half, then the program change that commits them.
    assert link.sent == [[0xB0, 0, 0x08], [0xB0, 32, 0x02], [0xC0, 0x50]]


def test_master_tune_reads_zero_cents_at_the_stored_centre():
    link = FakeLink({(0x40, 0x00, 0x00): [0x00, 0x04, 0x00, 0x00]})
    assert parts.master_tune_cents(link, device_id=0x10) == 0.0


def test_master_tune_is_none_when_the_address_will_not_answer():
    assert parts.master_tune_cents(FakeLink({}), device_id=0x10) is None
