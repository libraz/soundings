"""The model id, which decides what three address bytes name.

A unit answering under more than one model id opens a separate space in each,
and the same address in two of them is two different parameters. So there are
two things to get wrong and both are silent: asking in one space while recording
another, and accepting a reply that came back from a space nobody asked in.

The second is the one that produces a measurement. A reply carrying the right
address under the wrong model id parses into a value of exactly the right shape,
and laid into a record it is indistinguishable from a byte the asked-for address
held. Every reader that checks an address is therefore made to check this too,
and each check is exercised here against a unit that answers only in the space
it was built for.

No hardware. The fakes stand in for the link, not for any machine.
"""

from __future__ import annotations

from soundings import roland, window, writeback
from soundings.aliases import Snapshotter
from soundings.cli import build_parser, main

ADDRESS = (0x10, 0x00, 0x00)
"""Three bytes that name something in more than one space, which is the whole case."""

OTHER = 0x45


class OneSpace:
    """A unit that answers every read, always under the one model id it was given."""

    def __init__(self, model_id: int = roland.GS_MODEL_ID, memory=(0x11, 0x22, 0x33, 0x44)):
        self.model_id = model_id
        self.memory = list(memory)
        self.sent: list[list[int]] = []

    def send(self, message: list[int]) -> None:
        self.sent.append(list(message))

    def exchange(self, request: list[int], timeout: float = 0.0) -> list[int]:
        if len(request) < 11 or request[4] != roland.CMD_RQ1:
            return []
        address = (request[5], request[6], request[7])
        size = (request[8] << 14) | (request[9] << 7) | request[10]
        return roland.dt1(address, self.memory[:size], model_id=self.model_id)

    def receive(self, timeout: float = 0.0) -> list[int]:
        return []

    def drain(self) -> None:
        return None


def test_a_request_carries_the_model_id_it_was_given() -> None:
    assert roland.rq1(ADDRESS, 1)[3] == roland.GS_MODEL_ID
    assert roland.rq1(ADDRESS, 1, model_id=OTHER)[3] == OTHER
    assert roland.dt1(ADDRESS, [0x00], model_id=OTHER)[3] == OTHER


def test_a_reply_reports_the_space_it_came_from() -> None:
    """Reported rather than checked in the parser: the check belongs to the reader.

    A reader aimed at one space wants a foreign reply refused, and a run
    establishing which spaces a unit answers in wants to see it. Deciding here
    would leave the second unable to ask.
    """
    parsed = roland.parse_dt1(roland.dt1(ADDRESS, [0x7F], model_id=OTHER))
    assert parsed is not None
    assert parsed.model_id == OTHER
    assert parsed.data == [0x7F]


def test_a_snapshot_of_one_space_does_not_take_another_space_bytes() -> None:
    """A capture is compared byte for byte against a later one, so a foreign byte
    lands in a record as a value the asked-for address held."""
    unit = OneSpace(model_id=roland.GS_MODEL_ID)
    asked_where_it_answers = Snapshotter(unit, [(ADDRESS, 4)], timeout=0.0)
    assert len(asked_where_it_answers.take()) == 4

    asked_elsewhere = Snapshotter(unit, [(ADDRESS, 4)], model_id=OTHER, timeout=0.0)
    assert asked_elsewhere.take() == {}
    assert asked_elsewhere.unread == 1
    # Unread, not short: the region answered nothing in the space that was asked,
    # which is a different absence from a reply that arrived and was too small.
    assert asked_elsewhere.short == []


def test_a_write_probe_reads_back_only_in_the_space_it_wrote_in() -> None:
    """The whole probe is write-then-read-back, so a foreign read-back would
    report a range for an address in a space nothing was written to."""
    unit = OneSpace()
    assert writeback.Writer(unit, read_timeout=0.0).read_byte(ADDRESS) == 0x11
    assert writeback.Writer(unit, model_id=OTHER, read_timeout=0.0).read_byte(ADDRESS) is None


def test_a_window_probe_reads_only_in_the_space_it_is_aimed_at() -> None:
    unit = OneSpace()
    assert window.Prober(unit, read_timeout=0.0).read(ADDRESS) == 0x11
    assert window.Prober(unit, model_id=OTHER, read_timeout=0.0).read(ADDRESS) is None


def test_a_sweep_does_not_map_a_region_of_a_space_it_did_not_ask_in() -> None:
    """A hit is what puts a region on the map, and the map aims every later stage."""
    from soundings.sweep import Sweeper, Timing

    timing = Timing(base_ms=1.0, per_byte_ms=0.0, margin=1.0)
    assert Sweeper(OneSpace(), timing=timing, canary=ADDRESS).answers(ADDRESS)
    assert not Sweeper(OneSpace(), timing=timing, model_id=OTHER, canary=ADDRESS).answers(ADDRESS)


def test_a_stage_that_does_not_take_a_model_id_refuses_one_rather_than_sending(capsys) -> None:
    """It refuses before the port is opened, so nothing reaches the unit.

    A GS Reset is a write to one address in one space; sent under another model
    id it is a write to whatever those bytes name there. Running the stage and
    noting the model id in its record would publish the reading of a space the
    reset never touched.
    """
    # Before the subcommand, as --port and --device-id are: the three of them say
    # what the run is addressed to, which is not a parameter of any one stage.
    assert main(["--model-id", "0x45", "tone-map"]) == 1
    said = capsys.readouterr().out
    assert "tone-map" in said and "45" in said


def test_the_stages_that_need_only_a_read_and_a_write_take_one() -> None:
    """The list is the claim that these stages know nothing of a family.

    Asserted against the parser rather than written out again here: a second copy
    of the list is a second thing to keep in step, and it is the recorded command
    line surface that holds the reviewed one.
    """
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    takes = {n for n, p in sub.choices.items() if p._defaults.get("takes_model_id")}
    assert {"sweep", "offsets", "power-on", "write-probe", "window-probe", "hold-probe"} <= takes
    assert not takes & {"reset-probe", "tone-map", "efx-map", "alias-scan", "efx-params"}


def test_a_record_says_which_space_the_run_asked_in(tmp_path) -> None:
    """Recorded beside the device id, because it is the other half of what makes
    three address bytes mean something."""
    from soundings import record

    record.invoked(record.Invocation(stage="sweep", midi_device_id="10", midi_model_id="45"))
    try:
        stamp = record.envelope({}, out_path=tmp_path / "any.json")["record"]
    finally:
        record.invoked(None)
    assert stamp["midi_model_id"] == "45"
    assert stamp["midi_device_id"] == "10"
