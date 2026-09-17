"""Flags several commands share, and the converters that read them.

Only the flags that are genuinely the same everywhere are here. `--settle` and
`--values` mean something different in each command that takes them and carry
their own defaults and help there; collapsing those would replace a true
description with a vague one, which costs more than the repetition saves.
"""

from __future__ import annotations

import argparse

SAVE_HELP = (
    "directory to keep every take in, as float WAV with a manifest. Device time is the "
    "scarce thing here and a verdict thrown away with its audio has to be re-recorded to "
    "be asked anything else; kept takes can be measured again with the machine unplugged"
)

LEAD_IN_HELP = (
    "dBFS the lead-in must stay under; above this something was sounding before the note and "
    "the noise floor, which is the yardstick for everything else, is wrong. Absolute rather "
    "than relative to the take, or a quiet stimulus reads the same as a contaminated one"
)


def number(text: str) -> int:
    """An int in any base the prefix names, so 0x40 and 64 are both accepted."""
    return int(text, 0)


def pair(text: str) -> tuple[int, ...]:
    """'0,127' -- the two values a stimulus is sent at."""
    return tuple(int(v, 0) for v in text.split(","))


def write_spec(text: str) -> tuple[str, tuple[int, ...]]:
    """'40 41 22=01' or '40 03 00=02 01' -- an address and the bytes to put there.

    Several bytes because a parameter wider than one is not reachable a byte at
    a time: the insertion effect's type ignores a single byte outright, and a
    preparation that quietly did nothing turns the run it was setting up into a
    null about the parameter instead of about the state.
    """
    address, _, values = text.partition("=")
    address = address.strip()
    if not address or not values.strip():
        raise argparse.ArgumentTypeError(f"expected 'ADDRESS=BYTE [BYTE...]', got {text!r}")
    return address, tuple(int(v, 16) for v in values.split())


def bank_program(spec: str) -> tuple[int, int]:
    """'80' or '8:80' -- a bare number is the GM bank, which is bank 0."""
    bank, _, program = spec.rpartition(":")
    return (int(bank or 0, 0), int(program, 0))


def add_out(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", help="write the result as JSON")


def add_subject(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
    slot: bool = True,
    controller: bool = False,
) -> None:
    """What the takes were of, so the record says it in a field and not in its name.

    This one is genuinely the same everywhere, which is the bar this module sets:
    a type is two hex bytes and a slot is an address, on every stage that reads a
    parameter setting by setting. It is here rather than repeated because a stage
    that spelled it differently would file its records under a subject no query
    reaches, and that is not a style complaint -- a unit's listing groups records
    by subject, and a record naming none is one no coverage figure can count.

    **What was swept is not always an address.** A stage in the chain is reached by
    the part's own controllers as well as by the type's parameters, and a run that
    sweeps one of those is about the controller: filing it under the address it
    happened to hold would say the run was about a parameter nobody moved. Where
    both are offered only one may be given, which is the guard -- one run swept one
    thing.

    `controller` is off by default, and that is a statement about the stage rather
    than about the flag. A stage offering it has to be able to put a controller in
    the record it writes, and one that cannot would take the flag and file the run
    under nothing. Which stages cannot is a gap in those stages; it is left where a
    reader can see it instead of being covered by a flag that does nothing.

    `slot` is off for the few stages that read a whole type rather than one of its
    parameters, and it takes the controller with it: a stage with nothing swept
    inside the type has no controller subject either. Optional by default because
    the older stages have records that were made before the flag existed, and a
    flag made required today would make those commands unable to re-publish the
    very records that need it.
    """
    parser.add_argument(
        "--type",
        metavar="MSB LSB",
        required=required,
        help="the insertion effect type the takes were made under. Without it the "
        "record says what a parameter did without saying whose parameter it was, and "
        "a directory of them is identified by its filenames -- which is an index kept "
        "by hand beside records that could carry it themselves",
    )
    if slot:
        swept = parser.add_mutually_exclusive_group(required=required)
        swept.add_argument(
            "--slot",
            metavar="ADDR",
            help="the address that was swept, for the same reason",
        )
        if controller:
            swept.add_argument(
                "--cc",
                type=int,
                metavar="N",
                help="the controller that was swept, where the run reached the chain "
                "through one instead of through an address. Spelled as the field the "
                "index already groups on, so a run about a controller meets the runs "
                "that sent one rather than sitting in a group of its own",
            )


SUBJECT_IN_A_MANIFEST = ("type", "address", "controller")
"""The fields a store may close over to say what its takes were of.

Named here and not in the store, because this is the half that reads them and a
list of field names kept twice is two lists.
"""


def said_by(manifest: dict | None) -> dict[str, str]:
    """What a run's own store said its takes were of, and nothing a flag said.

    A run records its subject beside its takes, so a reading made from them can say
    what it is of rather than be told again. Read from the manifest alone, unlike
    `subject_of`, and the difference is not a shortcut: a record binding six runs
    across three types has no one subject it could be told, and a flag covering the
    whole directory would put the first run's type on all six. Where the store said
    nothing this returns nothing, which is a run that was never told and reads as
    one.
    """
    said = manifest or {}
    found = {key: said.get(key) for key in SUBJECT_IN_A_MANIFEST}
    # The two are exclusive in the same way the flags are: a run reached through a
    # controller was not reached through an address, and carrying both would put a
    # run in two groups of an index that means them to be one.
    if found["controller"] is not None:
        found["address"] = None
    return {key: value for key, value in found.items() if value not in (None, "")}


ABOUT_IS_NOT = frozenset({"takes", "stimuli", "windows_s", "question"})
"""Manifest keys a record that binds runs already carries from the reading itself.

The takes are the listing, the stimulus comes back as a name on the run, the windows
are a figure the reading publishes, and the question is prose the record asks once at
its top rather than once per run.

**Everything else a manifest holds is something the run said about itself, and it is
carried across by not being named here rather than by being listed.** A run that
states one more control than the run before it must not lose it to a reader written
against the earlier one.
"""


def about_of(manifest: dict | None) -> dict:
    """What one run said about itself, as the field a record puts beside that run.

    A record that binds several runs has one command line and several subjects, so
    what each run was of can only come from the run. This is the whole of the store's
    account of it -- the type, the address, what was held, whatever else the driver
    closed over -- minus the few things the reading publishes itself.

    Empty values are dropped rather than written: a run that states no held block and
    one that states an empty one are different, and a reader shown `held: null`
    cannot tell which it has.
    """
    said = manifest or {}
    return {
        key: value
        for key, value in said.items()
        if key not in ABOUT_IS_NOT and value not in (None, "", [], {})
    }


def subject_of(args: argparse.Namespace, manifest: dict | None = None) -> dict[str, str]:
    """The subject flags as the fields a record carries, dropping what was not given.

    The other half of `add_subject`, and here rather than in each handler so that
    the flag and the field cannot be spelled differently: a record filed under
    `slot` where its siblings say `address` is a record the unit's listing groups
    on its own, which reads as a parameter nobody measured.

    **The run's own manifest answers where the command line does not.** A driver
    that saved its takes wrote what they were of into the manifest beside them, and
    a reading made from those takes is a reading of that -- so a flag is a way of
    saying it where the takes do not, not the only way. The flag still wins: it is
    the one a person typed while looking at the run, and a manifest naming a whole
    directory of runs names the first of them.

    A field left out rather than written empty. An index shows an empty subject as
    a record that states none, which is true of a record that was never told and
    false of one told nothing.
    """
    said = said_by(manifest)
    given = getattr(args, "cc", None)
    found = {
        "type": getattr(args, "type", None) or said.get("type"),
        "address": getattr(args, "slot", None) or said.get("address"),
        # A controller number may be nought, which is a controller and not an
        # absence, so this one is asked whether it was given rather than whether it
        # is true. The flags are exclusive, so a run that named a controller does
        # not fall back to an address the manifest carried for another reason.
        "controller": given if given is not None else said.get("controller"),
    }
    if found["controller"] is not None:
        found["address"] = None
    return {key: value for key, value in found.items() if value not in (None, "")}


def add_audio(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audio", help="substring of the audio input device name")


def add_channel(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--channel", type=int, default=0, help="zero based MIDI channel")


def add_verify_reads(
    parser: argparse.ArgumentParser, *, default: int = 20, help: str | None = None
) -> None:
    parser.add_argument("--verify-reads", type=int, default=default, help=help)


def add_save(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--save", help=SAVE_HELP)


def add_max_lead_in(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-lead-in", type=float, default=-60.0, help=LEAD_IN_HELP)
