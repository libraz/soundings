"""Rolling per-address records up into one statement about a block.

Three commands over records that other stages left. One decides which pair of
values each address in a block should be asked at, and the other two read the
resulting per-address verdicts back into a single answer -- about a block of the
address space, or about the parameters of one insertion effect type.

The plan is the authority in both directions: it says how many addresses there
were to ask, so a short set of verdicts reads as incomplete rather than as
complete, and a record it does not name is left out rather than counted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import documents
from . import options, report

BAND_SET_NAMES = ("third-octave", "twelfth-octave")
"""The band sets `efx-bands` offers by name, spelled here so the parser can be
built without loading the reader. `efxbands.BAND_SETS` is what they resolve to,
and a test holds the two lists against each other rather than a reader doing it."""

ORDERS_READ = 8
"""How many orders `efx-orders` reads where the invocation does not say, spelled
here for the same reason and held against `efxorders.ORDERS` by the same test."""


def register(sub) -> None:
    p = sub.add_parser(
        "plan",
        help="say what pair of values each address in a block should be asked at, "
        "from what the write probe measured it to accept",
    )
    p.add_argument("write_probe", help="a write-probe record covering the block")
    p.add_argument(
        "block",
        help="the leading bytes of the addresses to plan, e.g. '40 11' for part 1",
    )
    p.add_argument(
        "--states-from",
        help="an effect-list document, to ask a parameter printed as a list of states at two "
        "of its own states instead of at the ends of what its address accepts. An address "
        "accepting values its parameter has no state for reads as a parameter that does "
        "nothing, and a block accepting every value at every address never clamps to show it",
    )
    p.add_argument(
        "--type",
        help="which effect's parameters to read out of --states-from, as two hex bytes",
    )
    p.add_argument(
        "--first-parameter",
        help="the address the type's first printed parameter sits at, e.g. '40 03 03'. Stated "
        "rather than assumed: where a block's parameters begin is a fact about a family and "
        "this command is given one unit's probe",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_plan)

    p = sub.add_parser(
        "block",
        help="read a directory of per-address contrast records into one verdict per "
        "address, counted against the plan the block was asked from",
    )
    p.add_argument("plan", help="the plan the block was asked from, which says how many")
    p.add_argument(
        "plain",
        nargs="?",
        help="records from the pass that asked the plain note. Omitted only for a block "
        "whose plan asks nothing, where there is no pass to point at",
    )
    p.add_argument(
        "--gesture",
        help="records from the pass that moved messages after the setting. A gesture "
        "is a rescue for a null, so it answers only where the plain note could not",
    )
    p.add_argument(
        "--polyphony",
        help="records from the pass that played two notes. A parameter about polyphony "
        "sounds identical under a single note however it is set, so an address left null "
        "by the one-note passes has not been asked rather than answered",
    )
    p.add_argument(
        "--superseded-by",
        action="append",
        default=[],
        metavar="RECORD",
        help="a published record that asked some of these addresses and disagrees. Each such "
        "address is named with that record and the verdict it carries there, so a reader "
        "landing on this file does not take an answer the archive has since bettered. Only "
        "disagreements are marked",
    )
    p.add_argument(
        "--balance",
        action="append",
        default=[],
        metavar="RECORD",
        help="a balance record, once per pass. A parameter that moves signal between the "
        "channels is invisible to a comparison made in one of them, so what it found is "
        "folded in as another way of having reached the signal path. Each pass has its own "
        "takes and its own balance, so give the plain pass's and the gesture pass's both",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_block)

    p = sub.add_parser(
        "efx-rate",
        help="read what one insertion effect's rate slot modulated at, setting by "
        "setting, from takes already saved, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="a directory of takes with the takes-manifest.json a --save run wrote",
    )
    options.add_subject(p)
    p.add_argument(
        "--setting",
        required=True,
        metavar="REGEX",
        help="a pattern over each take's setting with a group named `value`, which is "
        "the byte it was taken at, or named `type` under --untouched. A run names its "
        "takes however its own question needed, so how the setting is read back out "
        "belongs in the invocation -- where it lands in the record, and a reader can "
        "check it rather than trust it. A second group named `from` may say what the "
        "address held, and the modulation had reached, when that byte was written "
        "with no reset between -- which is the only thing that tells two takes of one "
        "byte apart where the byte is one a modulator has to travel to",
    )
    p.add_argument(
        "--untouched",
        action="store_true",
        help="the takes were made with a type loaded and no parameter written, so there "
        "is no byte and the pattern names the type instead. This is the baseline a "
        "sweep of the same type is read against: a sweep says what changed with the "
        "byte and cannot say what was already there",
    )
    p.add_argument(
        "--byte-names-no-rate",
        action="store_true",
        help="the byte swept is not a rate slot, so what comes back is what the output "
        "was measured to repeat at rather than what the byte asked for. The reading is "
        "the same and what may be concluded from it is not: a rate slot returning its "
        "own setting is the slot answering, while a byte that names no rate returning a "
        "periodicity at all is a fact about the structure behind it. Which of the two a "
        "record is belongs to the run, because nothing in the numbers says it",
    )
    p.add_argument(
        "--held",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="an address the run had written while it read, and what it held. A type "
        "with two modulators returns whichever dominates, so a reading taken with the "
        "other stage turned down is a different reading and nothing in the number says so",
    )
    p.add_argument(
        "--held-not-spelled-out",
        default=None,
        metavar="TEXT",
        help="something the run held that has no address to give --held. A take name "
        "may say a state -- the type's other rate slot at some byte -- without saying "
        "which address that is, and which address it is depends on the type. Said in "
        "words rather than guessed at, because a record stating an address that was "
        "never written is wrong in a way a reader cannot see, and one stating nothing "
        "held reads as a run that held nothing",
    )
    p.add_argument(
        "--settled",
        type=float,
        help="seconds the run waited after writing the setting before recording. One "
        "family accelerates for about four seconds and a take begun before that returns "
        "the ramp's average, so a run that did not wait records that it did not",
    )
    p.add_argument(
        "--lead", type=float, default=0.6, help="seconds of silence at the head of a take"
    )
    p.add_argument(
        "--hold",
        type=float,
        help="seconds the note was held, where the manifest's own take length is not "
        "one second longer than it",
    )
    p.add_argument(
        "--lines",
        action="store_true",
        help="also report the lines the partials' level spectra share. A vote returns "
        "one answer and lands between two modulators; this returns both, which is what "
        "a type with a modulator per stage needs",
    )
    p.add_argument(
        "--channel",
        type=int,
        metavar="N",
        help="the interface channel to read every take from. Defaults to whichever "
        "reached highest across the takes read, chosen once for the run: an input the "
        "unit is not on is not silent, so reading each take's own loudest channel gives "
        "a setting that silenced the output a level and a rate belonging to that input",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_rate)

    p = sub.add_parser(
        "efx-sway",
        help="read what one insertion effect's modulator did to level and to balance, "
        "setting by setting, from takes already saved, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="a directory of takes with the takes-manifest.json a --save run wrote",
    )
    options.add_subject(p, required=True)
    p.add_argument(
        "--setting",
        required=True,
        metavar="REGEX",
        help="a pattern over each take's name with a group named `value`, which is the "
        "byte it was taken at. A run names its takes however its own question needed, so "
        "how the setting is read back out belongs in the invocation -- where it lands in "
        "the record, and a reader can check it rather than trust it",
    )
    p.add_argument(
        "--still",
        metavar="REGEX",
        help="a pattern naming the take with no modulation in it -- the modulator "
        "switched off, or the part routed past the effect -- which the control is "
        "injected into. Named rather than guessed: a take already carrying a modulation "
        "ends up with two, the search finds the unit's own, and the control reads as "
        "having failed on material it can read perfectly well",
    )
    p.add_argument(
        "--held",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="an address the run had written while it read, and what it held. A type "
        "with more than one modulator returns whichever dominates, so a reading taken "
        "with the other stage held still is a different reading and nothing in the "
        "number says so",
    )
    p.add_argument(
        "--lead", type=float, default=0.6, help="seconds of silence at the head of a take"
    )
    p.add_argument("--min-rate", type=float, default=0.3, help="slowest modulation searched, Hz")
    p.add_argument("--max-rate", type=float, default=15.0, help="fastest modulation searched, Hz")
    p.add_argument(
        "--least",
        type=float,
        default=0.5,
        help="dB peak to peak a cycle must reach before a rate is reported. Under this a "
        "held note's own unsteadiness fits a slow oscillation as well as a modulator does, "
        "and the control is what says whether a row over the bar is worth reading",
    )
    p.add_argument(
        "--channels",
        type=int,
        nargs=2,
        metavar="N",
        help="the interface channels to read every take from. Defaults to the pair that "
        "reached highest across the takes read, chosen once for the run: half of what "
        "this reads is the difference between two channels, and a difference taken "
        "across a pair chosen per take is a difference between two different things",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_sway)

    p = sub.add_parser(
        "efx-bands",
        help="read what one insertion effect's parameter did to the level of each third "
        "octave, setting by setting, from takes already saved, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="a directory of takes with the takes-manifest.json a --save run wrote",
    )
    options.add_subject(p, required=True)
    p.add_argument(
        "--setting",
        required=True,
        metavar="REGEX",
        help="a pattern over each take's setting with a group named `value`, which is "
        "the byte it was taken at. A run names its takes however its own question "
        "needed, so how the setting is read back out belongs in the invocation -- where "
        "it lands in the record, and a reader can check it rather than trust it",
    )
    p.add_argument(
        "--reference",
        required=True,
        metavar="REGEX",
        help="a pattern naming the repeats of the setting the run held flat. Every "
        "profile is reported against their mean and has to clear their spread, and they "
        "come from this directory because what a stimulus fails to repeat belongs to the "
        "session it was recorded in",
    )
    p.add_argument(
        "--control",
        metavar="REGEX",
        help="a pattern naming the takes made with the part routed past the effect. "
        "Without it the record cannot say whether the flat setting was the effect doing "
        "nothing, and every deviation below would imply a unity it never measured",
    )
    p.add_argument(
        "--silence",
        metavar="REGEX",
        help="a pattern naming the takes made with the same chain and nothing played. "
        "A setting that turns the output down far enough returns the room and the "
        "converter, and without this the floor's own shape is published as a profile",
    )
    p.add_argument(
        "--stimulus",
        help="what was sounded through the effect. A band a stimulus does not reach "
        "cannot report what the effect did there, so which stimulus carried the take "
        "bounds the record rather than decorating it",
    )
    p.add_argument(
        "--held",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="an address the run had written while it read, and what it held. A band "
        "profile is the whole chain's, so a parameter read with another of the type's "
        "stages moved is a reading of something else and nothing in the numbers says so",
    )
    p.add_argument(
        "--reference-held",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="an address the reference takes were made under, where that differs from "
        "the sweep. The reference is a state and not one of the readings, and on a "
        "printed range whose last position is the stage switched out it is a value of "
        "the byte being swept -- so without this the record reports every profile "
        "against something it cannot name",
    )
    p.add_argument(
        "--band",
        type=float,
        action="append",
        metavar="HZ",
        help="a band centre to measure, repeatable. Defaults to third octaves from 100 "
        "to 12500, because an octave band cannot separate two corners printed one octave "
        "apart",
    )
    p.add_argument(
        "--band-set",
        choices=BAND_SET_NAMES,
        default="third-octave",
        help="how fine to read, where --band is not given. A band wider than the "
        "deviation in it reports that deviation shallower than it was, and a deviation "
        "still growing where the set ends is reported as though it had stopped there; "
        "twelfth-octave is four times finer and reaches two octaves lower, over the "
        "same takes",
    )
    p.add_argument(
        "--channel",
        type=int,
        metavar="N",
        help="the interface channel to read every take from. Defaults to whichever is "
        "loudest in the reference takes, chosen once for the run: an input the unit is "
        "not on is not silent, so reading each take's own loudest channel makes a "
        "setting that turns the output down a full profile of something else",
    )
    p.add_argument(
        "--lead", type=float, default=0.6, help="seconds of silence at the head of a take"
    )
    p.add_argument(
        "--hold",
        type=float,
        help="seconds the stimulus was held, where the manifest's own take length is not "
        "one second longer than it",
    )
    p.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("OPENS", "WIDE"),
        help="read the bands over one stretch of each take, named in seconds from the "
        "start of it, rather than over the whole of the held stimulus. For a question "
        "about how a profile changes over one sounding -- an effect that builds, a tail "
        "that darkens -- where the answer is two windows of the same takes put side by "
        "side, which is two records and a derivation between them. Reported beside the "
        "hold rather than in place of it, because the stimulus was held as long as it "
        "was held",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_bands)

    p = sub.add_parser(
        "efx-orders",
        help="read what one setting did to the harmonic orders of a held tone, order by "
        "order, from takes already saved, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="a directory of takes with the takes-manifest.json a --save run wrote",
    )
    options.add_subject(p, required=True, controller=True)
    p.add_argument(
        "--setting",
        required=True,
        metavar="REGEX",
        help="a pattern over each take's setting with a group named `value`, which is "
        "what it was taken at. The same contract the band reading sets: how the setting "
        "is read back out belongs in the invocation, where it lands in the record and a "
        "reader can check it rather than trust it",
    )
    p.add_argument(
        "--carrier",
        type=float,
        required=True,
        metavar="HZ",
        help="where the carrier was nominally played. Each take's own is measured near "
        "it and reported, because a boxcar aimed off the partial reads every order "
        "through the skirt of its own window",
    )
    p.add_argument(
        "--orders",
        type=int,
        default=ORDERS_READ,
        metavar="N",
        help="how many orders to read, counting the first. A count and not a bound on "
        "the stage: what the curve put above the last one asked for is not in the "
        "figures, and the record says how much level the series still carried there",
    )
    p.add_argument(
        "--over-periods",
        type=int,
        metavar="N",
        help="read each order through a boxcar this many carrier periods long instead of "
        "over the whole held stretch. A resolution and not a correction: the short "
        "filter's lobe is wide enough to count what the stage put beside an order as "
        "part of it, and the long one loses a partial that moves. The same takes read "
        "both ways are two records, and two that disagree say where the energy sits",
    )
    p.add_argument(
        "--reference",
        metavar="REGEX",
        help="a pattern naming the repeats of one setting. Optional here and required "
        "on the band reading, because an order is read under its own take's first order "
        "rather than against these: what the repeats add is the floor, so a run without "
        "them is short of a bound rather than short of a reading",
    )
    p.add_argument(
        "--control",
        metavar="REGEX",
        help="a pattern naming the takes made with the part routed past the effect. They "
        "are what says an order belongs to the stage rather than to the voice, which a "
        "carrier that is not one partial arrives with",
    )
    p.add_argument(
        "--silence",
        metavar="REGEX",
        help="a pattern naming the takes made with the same chain and nothing played. An "
        "order that has fallen into the room and the converter is the floor being "
        "reported as a harmonic",
    )
    p.add_argument(
        "--stimulus",
        help="what was sounded through the effect. How much of it is its own fundamental "
        "is what bounds this reading, because a curve fed two partials returns products "
        "between them and a product on a whole multiple is counted as that order",
    )
    p.add_argument(
        "--held",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="an address the run had written while it read, and what it held. Orders are "
        "the whole chain's, so a setting read with another of the type's stages moved is "
        "a reading of something else",
    )
    p.add_argument(
        "--channel",
        type=int,
        metavar="N",
        help="the interface channel to read every take from. Defaults to whichever is "
        "loudest in the takes the run repeated, or in the swept takes where it repeated "
        "none: an input the unit is not on is not silent, and its orders are an answer",
    )
    p.add_argument(
        "--lead", type=float, default=0.6, help="seconds of silence at the head of a take"
    )
    p.add_argument(
        "--hold",
        type=float,
        help="seconds the stimulus was held, where the manifest's own take length is not "
        "one second longer than it",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_orders)

    p = sub.add_parser(
        "efx-time",
        help="read where one insertion effect's delay slot put the copy it returns, "
        "setting by setting, from takes already saved, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="a directory of takes with the takes-manifest.json a --save run wrote",
    )
    options.add_subject(p, required=True)
    p.add_argument(
        "--setting",
        required=True,
        metavar="REGEX",
        help="a pattern over each take's setting with a group named `value`, which is "
        "the byte it was taken at. Two takes matching one value is how this stage gets "
        "its floor, so a run that repeated a setting needs no separate pattern for it",
    )
    p.add_argument(
        "--byte-names-no-delay",
        action="store_true",
        help="the byte swept is not a delay slot, so what comes back is the distance "
        "the output was measured to hold a copy of itself at rather than a time the "
        "byte asked for. It also moves what the two copies are: a delay run reads the "
        "source against its own return, while a run made with the source out of the "
        "mixture reads whatever the effect's output holds twice. Which of the two a "
        "record is belongs to the run, because nothing in the numbers says it",
    )
    p.add_argument(
        "--control",
        required=True,
        metavar="REGEX",
        help="a pattern naming the takes made with the part routed past the effect. "
        "Required rather than optional here: their cepstrum is subtracted from every "
        "reading, so without them each row would carry whatever the stimulus and the "
        "converters put into a log spectrum and report it as a copy",
    )
    p.add_argument(
        "--stimulus",
        help="what was sounded through the effect. The reading wants a source whose own "
        "log spectrum is smooth, and a held note's is a comb of its own, so which "
        "stimulus carried the take bounds the record rather than decorating it",
    )
    p.add_argument(
        "--held",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="an address the run had written while it read, and what it held. A time "
        "read out of a cepstrum is a time through the whole chain, so a feedback path "
        "left open returns further copies and a balance carrying only the return leaves "
        "nothing for the copy to beat against and no peak at all",
    )
    p.add_argument(
        "--frame",
        type=int,
        default=65536,
        help="samples per cepstral frame. It bounds the reading at both ends -- half of "
        "it is the longest delay that can be placed, and a delay that moves inside one "
        "smears until nothing stands out -- so a longer printed range needs a longer "
        "frame and buys it with fewer frames to average",
    )
    p.add_argument("--hop", type=int, default=16384, help="samples between frames")
    p.add_argument(
        "--shortest",
        type=float,
        default=None,
        help="milliseconds below which no peak is read. Defaults to the transform's own "
        "resolution; set higher and a delay under it comes back as its third rahmonic, "
        "which stands as high as a delay and is not one",
    )
    p.add_argument(
        "--longest",
        type=float,
        default=620.0,
        help="milliseconds above which no peak is read. Set past the printed end of the "
        "range, so a byte that runs further than the page says shows as a reading rather "
        "than as the search's own edge",
    )
    p.add_argument(
        "--peaks-apart",
        type=float,
        default=None,
        help="milliseconds set aside either side of a peak before the next is taken. "
        "Defaults to the transform's own resolution, which is what a peak is wide; a "
        "window set wider closes over the peaks that say a reading sits on a rahmonic",
    )
    p.add_argument(
        "--channel",
        type=int,
        metavar="N",
        help="the interface channel to read every take from. Defaults to whichever is "
        "loudest in the takes with the effect out, chosen once for the run: an input the "
        "unit is not on is not silent, and a take read from one returns a full plausible "
        "cepstrum of something else",
    )
    p.add_argument(
        "--lead", type=float, default=0.6, help="seconds of silence at the head of a take"
    )
    p.add_argument(
        "--hold",
        type=float,
        help="seconds the stimulus was held, where the manifest's own take length is not "
        "one second longer than it",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_time)

    p = sub.add_parser(
        "efx-excursion",
        help="read how far one insertion effect's modulator swings the delay under it, "
        "setting by setting, from takes already saved, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="a directory of takes with the takes-manifest.json a --save run wrote",
    )
    options.add_subject(p, required=True)
    p.add_argument(
        "--setting",
        required=True,
        metavar="REGEX",
        help="a pattern over each take's setting with a group named `value`, which is "
        "the byte it was taken at. Two takes matching one value is how this stage gets "
        "its floor, so a run that repeated a setting needs no separate pattern for it",
    )
    p.add_argument(
        "--carrier-hz",
        type=float,
        required=True,
        help="the fundamental of the note the takes hold. Its orders are what the "
        "reading is taken on, and the demodulator's window is one period of it, which "
        "is what puts the neighbouring orders on a null",
    )
    p.add_argument(
        "--bypassed",
        default="bypassed",
        help="a pattern naming the takes made with the part routed past the effect. "
        "Which orders every reading is taken on is decided there and never by the take "
        "being read: an effect that adds sidebands raises its own quiet orders over the "
        "threshold and brings a different set of partials into its own answer",
    )
    p.add_argument(
        "--held",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="an address the run had written while it read, and what it held. An "
        "excursion is read through the whole chain, so a feedback path around the delay "
        "deepens the notch without moving it and a balance away from the point that "
        "carries both paths shallows the series the fit is taken on",
    )
    p.add_argument(
        "--held-not-spelled-out",
        default=None,
        metavar="TEXT",
        help="something the run held that has no address to give --held. Said in words "
        "rather than guessed at, because a record stating an address that was never "
        "written is wrong in a way a reader cannot see",
    )
    p.add_argument(
        "--settled",
        type=float,
        help="seconds the run waited after writing the setting before recording, for any "
        "type that takes time to reach the state it was asked for",
    )
    p.add_argument(
        "--lead", type=float, default=0.6, help="seconds of silence at the head of a take"
    )
    p.add_argument(
        "--hold",
        type=float,
        default=8.0,
        help="seconds the note was held for. Half a second is cut from each end of it, "
        "because an attack and a release are not the steady tone this reads",
    )
    p.add_argument("--min-rate", type=float, default=0.20, help="slowest rate the grid reaches, Hz")
    p.add_argument("--max-rate", type=float, default=8.0, help="fastest rate the grid reaches, Hz")
    p.add_argument(
        "--step-hz",
        type=float,
        default=0.01,
        help="how finely the grid is walked. This is not the resolution -- two rates "
        "closer than one over the length read are one peak however fine the grid is -- "
        "it is only how precisely that peak can be placed",
    )
    p.add_argument(
        "--channel",
        type=int,
        metavar="N",
        help="the interface channel to read every take from. Defaults to whichever "
        "reached highest across the takes read, chosen once for the run: the unit's two "
        "outputs carry a stereo modulator in antiphase, so a level series read on one is "
        "not the level series read on the other",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_excursion)

    p = sub.add_parser(
        "efx-pair",
        help="read what one insertion effect does between the unit's two outputs, "
        "setting by setting, from takes already saved, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="a directory of takes with the takes-manifest.json a --save run wrote",
    )
    options.add_subject(p, required=True)
    p.add_argument(
        "--setting",
        required=True,
        metavar="REGEX",
        help="a pattern over each take's setting with a group named `value`, which is "
        "the byte it was taken at. Two takes matching one value is how this stage gets "
        "the floor a difference between settings has to clear",
    )
    p.add_argument(
        "--bypassed",
        default=None,
        metavar="REGEX",
        help="a pattern naming the takes made with the part routed past the effect. Two "
        "of them are read against each other to give the null; one alone gives none, "
        "because a take read against itself returns nought by construction",
    )
    p.add_argument(
        "--held",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="an address the run had written while it read, and what it held. What "
        "stands between the two outputs is the whole chain, so a pan byte away from "
        "centre and a balance away from the point carrying both paths each change what "
        "a lag is a lag of",
    )
    p.add_argument(
        "--held-not-spelled-out",
        default=None,
        metavar="TEXT",
        help="something the run held that has no address to give --held",
    )
    p.add_argument(
        "--outputs-from",
        default=None,
        metavar="PATH",
        help="an output-map record for this rig, which says which capture channels each "
        "socket pair on the back of the unit arrived on. It does not say which channel "
        "of a pair is the left, and neither does this stage: what is read here is the "
        "lag between the two channels of one pair, in whichever order they were patched",
    )
    p.add_argument(
        "--settled",
        type=float,
        help="seconds the run waited after writing the setting before recording",
    )
    p.add_argument(
        "--lead", type=float, default=0.6, help="seconds of silence at the head of a take"
    )
    p.add_argument(
        "--trim",
        type=float,
        default=0.2,
        help="seconds cut from each end of the held note, because an attack and a "
        "release are not the steady tone this reads",
    )
    p.add_argument("--hold", type=float, default=8.0, help="seconds the note was held for")
    p.add_argument(
        "--block",
        type=int,
        default=None,
        help="samples each lag is read over. Short enough that a modulator moves between "
        "blocks rather than being averaged flat, which is what lets the control fire",
    )
    p.add_argument(
        "--band-set",
        choices=BAND_SET_NAMES,
        default="third-octave",
        help="how fine the phase between the two outputs is read. The same two sets the "
        "band stages offer, so a phase and a level profile of one effect are read on one "
        "set of bands and can be put beside each other",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_pair)

    p = sub.add_parser(
        "efx-params",
        help="read a directory of per-address records into one verdict per parameter of "
        "one insertion effect type, with no machine attached",
    )
    p.add_argument("records", help="a directory of contrast records, one per parameter address")
    options.add_subject(p, required=True, slot=False)
    p.add_argument(
        "--types-from", required=True, help="an efx-type-map record, for the settings it loads"
    )
    p.add_argument(
        "--control", required=True, help="the routed-against-bypassed record from the same run"
    )
    p.add_argument(
        "--prepare",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="the state the run was taken in, recorded with it since a verdict holds in it",
    )
    p.add_argument(
        "--supersede",
        action="append",
        default=[],
        metavar="ADDR=RECORD",
        help="an address whose first pair was withdrawn, and the record that replaced it. "
        "A pair that left one setting silent compares sound with silence, and the reason "
        "travels with the row rather than with whoever remembers the directory",
    )
    p.add_argument(
        "--slots",
        nargs="+",
        default=[],
        metavar="ADDR",
        help="the type's parameter addresses in slot order, which is what makes a missing "
        "record readable. Without them the slot is the position in the sorted file names, so "
        "an address the run could not measure renumbers every parameter after it and pairs "
        "each with the default of the slot before, and the count reports the type as having "
        "one parameter fewer than it has. Given rather than derived because which address is "
        "which slot is a fact about the unit",
    )
    p.add_argument(
        "--states-from",
        help="an effect-list document, so a null taken at a value the parameter has no state "
        "for is reported as a slot nobody asked rather than as one that answered nothing. It "
        "decides nothing about what was heard",
    )
    p.add_argument(
        "--first-parameter",
        help="the block those parameters' addresses sit in, for --states-from: the "
        "list prints the last byte of each and which block it belongs to is a fact "
        "about a family rather than about this type",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_params)


def _printed_values(path: str, type_id: str, block: str) -> dict[str, str]:
    """The value column an effect list prints against each of a type's parameters.

    Keyed by the whole address, which the page gives the last byte of: the block
    those bytes sit in is a fact about a family and is named by the caller, not
    carried here.

    The value column and not the setting column. A count of the names between the
    slashes in the setting column gets three kinds of parameter wrong -- a rotor's
    two speeds, which are the bytes 0 and 127 rather than 0 and 1; a gain, whose
    setting column prints decibels and no names while its values run from 52 to 76;
    and a damping frequency printed `315-8k/Bypass`, which has a slash and is a
    referral to a table of 128 entries.

    Rows a person read off the page are merged with the rows the parser cut, because
    the residue is where the enumerations concentrate: the printing runs a name into
    its setting exactly where the name is long and the setting is a list of states.
    """
    where = Path(path)
    rows = json.loads(where.read_text())["rows"]
    hand = where.parent / "by-hand.json"
    if hand.is_file():
        rows = rows + json.loads(hand.read_text())["tables"].get(where.stem, [])
    head = block.replace(" ", "").upper()
    wanted = type_id.replace(" ", "").upper()
    out: dict[str, str] = {}
    for row in rows:
        if "address_lsb" not in row or "values_hex" not in row:
            continue
        if (row["msb"] + row["lsb"]).upper() != wanted:
            continue
        out[f"{head[0:2]} {head[2:4]} {row['address_lsb']}"] = row["values_hex"]
    return out


#: The marks an effect list prints in front of a parameter's name. The document
#: reader cuts them off and files them as `printed_mark`, so a row extracted by the
#: parser no longer carries one; this is for the rows somebody read by hand, where
#: a mark is typed as it was printed and nothing checks it.
#:
#: What each mark says is which of the two effect controls can reach that parameter
#: -- a fact about the type and worth keeping, which is why it is kept rather than
#: dropped. What it must not do is sit inside the name: `stage_named` reads the
#: first word of a name as the tag saying which stage of a two-stage type the
#: parameter is in, and a mark left in front of it answers `+`, which matches no
#: other parameter's stage and so reports every gate in its own stage as absent.
_PRINTED_MARKS = "+#* "


def _printed_names(path: str, type_id: str, block: str) -> dict[str, str]:
    """What an effect list calls each of a type's parameters, keyed by address.

    The same rows `_printed_values` reads, taken from the other column. What the
    name is for is the grouping: a multi-stage type is printed with a short tag in
    front of every parameter of each stage, which is the only statement anywhere
    about which addresses are one place in the signal path.
    """
    where = Path(path)
    rows = json.loads(where.read_text())["rows"]
    hand = where.parent / "by-hand.json"
    if hand.is_file():
        rows = rows + json.loads(hand.read_text())["tables"].get(where.stem, [])
    head = block.replace(" ", "").upper()
    wanted = type_id.replace(" ", "").upper()
    out: dict[str, str] = {}
    for row in rows:
        if "address_lsb" not in row or not row.get("parameter"):
            continue
        if (row["msb"] + row["lsb"]).upper() != wanted:
            continue
        name = str(row["parameter"]).lstrip(_PRINTED_MARKS).strip()
        if name:
            out[f"{head[0:2]} {head[2:4]} {row['address_lsb']}"] = name
    return out


#: Where a referral in an effect list's value column points. A `*n` is a reference
#: inside the document that prints it, so it is resolved from that document's own
#: directory rather than from a flag -- a grid handed in from somewhere else would
#: be a different publisher's table answering this one's footnote.
CONVERSION_GRID = "value-conversion.json"


def _printed_settings(path: str, states: dict[str, str]) -> dict[str, dict[int, str]]:
    """For each address whose value column refers to the grid, what the grid prints.

    Empty where the document has no grid beside it, which is not a failure: a page
    read for its effect list alone still plans, and every address keeps the pair its
    measured range gives it.
    """
    grid = Path(path).parent / CONVERSION_GRID
    if not grid.is_file():
        return {}
    rows = json.loads(grid.read_text())["rows"]
    out = {}
    for address, cell in states.items():
        each = documents.settings_printed(cell, rows)
        if each:
            out[address] = each
    return out


def cmd_plan(args: argparse.Namespace) -> int:
    """Turn a write probe's measured ranges into the pair each address is asked at."""
    from .. import plan

    states = None
    if args.states_from:
        if not args.type:
            print("--states-from needs --type to know which of the list's rows are whose")
            return 1
        states = _printed_values(args.states_from, args.type, args.block)
        if not states:
            print(f"{args.states_from} prints no values against any parameter of {args.type!r}")
            return 1

    each = _printed_settings(args.states_from, states) if states else {}
    record = json.loads(Path(args.write_probe).read_text())
    asks, skipped = plan.plan_block(record, args.block, states, each)
    if not asks and not skipped:
        print(f"no address under {args.block!r} in {args.write_probe}")
        return 1
    print(plan.summarise(asks, skipped))

    out = {
        "block": args.block,
        "from": str(args.write_probe),
        "method": plan.METHOD,
        "ask": [a.to_json() for a in asks],
        "cannot_be_asked": [s.to_json() for s in skipped],
    }
    if states is not None:
        out["states_from"] = {
            "document": str(args.states_from),
            "type": args.type,
            "block": args.block,
            "addresses": len(states),
            "referred_to_the_conversion_grid": sorted(each),
            "why_the_grid_is_read": (
                "A value column that refers to the grid narrows nothing, because the grid gives a "
                "setting at all 128 values. What it does answer is whether the pair chosen for the "
                "address names two settings or one of them twice, which a column that wraps does "
                "not: the one such column here prints each end as equal to the other, and a null "
                "read off that pair says the two values are one setting and not that the parameter "
                "does nothing."
            ),
        }
    report.write_json(args.out, out)
    return 0


def cmd_block(args: argparse.Namespace) -> int:
    """Read a block's per-address records into one answer about the block."""
    from .. import block

    planned = json.loads(Path(args.plan).read_text())
    plain, outside = (
        block.split_by_plan(block.survey(args.plain), planned) if args.plain else ({}, [])
    )
    gesture, more = (
        block.split_by_plan(block.survey(args.gesture), planned) if args.gesture else ({}, [])
    )
    outside += more
    polyphony, more = (
        block.split_by_plan(block.survey(args.polyphony), planned) if args.polyphony else ({}, [])
    )
    outside += more
    for address in sorted(set(outside)):
        print(f"  {address}: a record the plan does not name, left out of the block")
    # An empty pass is a mistake only where the plan expected one. A block whose
    # every address accepts a single value has no pair to compare and so no
    # contrast record to point at, and refusing to fold it would leave the one
    # thing measured about it -- that nothing there can be moved -- unpublished,
    # while the block went on being counted as a sweep somebody still owes.
    if not plain and not gesture and not polyphony and planned.get("ask"):
        print(f"no contrast records under {args.plain}")
        return 1

    found = [f for f in block.join(plain, gesture, polyphony) if f is not None]
    # One per pass rather than one for the block: each pass has its own takes and
    # its own balance, and a parameter that moves the balance under the gesture
    # is as much a finding as one that moves it under the plain note.
    for where in args.balance:
        found = block.with_balance(found, json.loads(Path(where).read_text()))
    found = block.with_the_plans_caveats(found, planned)
    # Only where the two actually disagree. A record that asked the same address
    # and answered the same way supersedes nothing, and marking it would tell a
    # reader to go elsewhere for the answer already in front of them.
    answered_elsewhere: dict[str, dict] = {}
    for where in args.superseded_by:
        other = json.loads(Path(where).read_text())
        mine = {f.address: f.verdict for f in found}
        for row in other.get("addresses", []):
            if row["address"] in mine and row["verdict"] != mine[row["address"]]:
                answered_elsewhere[row["address"]] = {
                    "record": Path(where).name,
                    "verdict": row["verdict"],
                }
    if answered_elsewhere:
        found = block.superseded(found, answered_elsewhere)
        print(f"  {len(answered_elsewhere)} addresses answered by another record")
    coverage = block.against_plan(found, planned)
    print(block.summarise(found, coverage))
    # Named rather than counted: an address left to try is the next run's list,
    # and a number is not a list.
    open_still = [f.address for f in found if f.still_open]
    if open_still:
        print(f"\nstill open: {' | '.join(open_still)}")

    report.write_json(
        args.out,
        {
            "block": planned.get("block"),
            "method": block.method_for(bool(args.gesture), asked=bool(planned.get("ask"))),
            # Where the addresses came from, which for a block nothing could be
            # asked of is the whole of its evidence: the record holds no verdict
            # of its own, and a reader has to be able to reach the run that
            # established there was nothing to ask.
            "planned_from": planned.get("from"),
            "asked_in": {
                name: found_states
                for name, where in (
                    ("plain", args.plain),
                    ("gesture", args.gesture),
                    ("polyphony", args.polyphony),
                )
                if where and (found_states := block.states(where))
            },
            "why_asked_in": block.WHY_ASKED_IN,
            **(
                {
                    "records_the_plan_does_not_name": sorted(set(outside)),
                    "why_outside_the_plan": block.WHY_OUTSIDE_THE_PLAN,
                }
                if outside
                else {}
            ),
            **({"two_passes": block.WHY_TWO_PASSES} if args.gesture else {}),
            **({"polyphony_pass": block.WHY_POLYPHONY_PASS} if args.polyphony else {}),
            **({"balance_counts": block.WHY_BALANCE_COUNTS} if args.balance else {}),
            "chose_the_values": planned.get("method"),
            "coverage": coverage,
            "still_open": open_still,
            "addresses": [f.to_json() for f in found],
        },
    )
    return 0


def cmd_efx_rate(args) -> int:
    """What one rate slot modulates at, read from takes already saved."""
    from .. import efxrate

    def said(reading) -> None:
        at = reading.get("value")
        head = f"{at:5d}" if at is not None else f"{reading['type']:>5s}"
        found = reading["rate_hz"]
        split = reading.get("split_between_hz")
        said_rate = (
            f"{found:8.4f} Hz" if found is not None
            else f"{'split':>8s}   " if split
            else f"{'--':>11s}"
        )
        print(
            f"  {head} -> {said_rate}"
            + f"  {reading['agreeing']}/{reading['of']} partials"
            f"  floor {reading['slowest_measurable_hz']}  {reading['heard_db']:.0f} dBFS"
            + (f"  between {' and '.join(f'{r:.4f}' for r in split)} Hz" if split else "")
        )

    if args.untouched and args.byte_names_no_rate:
        print("--untouched and --byte-names-no-rate are two different questions:")
        print("one has no byte at all, the other has a byte that is not a rate slot")
        return 2
    if args.untouched:
        if args.type or args.slot:
            print("--untouched takes no --type or --slot: nothing was written, and the")
            print("pattern names the type because that is what separates the takes")
            return 2
        found = efxrate.read_untouched(
            args.takes,
            setting=args.setting,
            settled_s=args.settled,
            lead_s=args.lead,
            hold_s=args.hold,
            shared_lines=args.lines,
            channel=args.channel,
            progress=said,
        )
    else:
        if not args.type or not args.slot:
            print("--type and --slot are required unless --untouched")
            return 2
        found = efxrate.read_directory(
            args.takes,
            type_id=args.type,
            address=args.slot,
            setting=args.setting,
            held=[{"address": a, "bytes": " ".join(f"{b:02X}" for b in v)} for a, v in args.held],
            held_not_spelled_out=args.held_not_spelled_out,
            settled_s=args.settled,
            names_a_rate=not args.byte_names_no_rate,
            lead_s=args.lead,
            hold_s=args.hold,
            shared_lines=args.lines,
            channel=args.channel,
            progress=said,
        )
    picked = found["channel"]
    print(
        f"  read from channel {picked['read']} of "
        f"{len(picked['reached_db'])} ({picked['chosen_by']}): "
        + " ".join(f"{v:.0f}" for v in picked["reached_db"])
        + " dBFS"
    )
    if not found["readings"]:
        print(
            f"no take under {args.takes} has a setting matching {args.setting!r}; "
            f"{found['takes_not_matching']['count']} were looked at"
        )
        return 1
    if missed := found["takes_not_matching"]["count"]:
        print(f"  ({missed} takes under the same directory did not match the pattern)")
    report.write_json(args.out, found)
    return 0


def _span_said(reading) -> str:
    """Where the deviation had halved, and whether it levelled off before the end.

    An open end is printed as such rather than left blank: a profile that never
    comes back down inside the set and one that comes back down at the last band
    are different answers, and a blank reads as the second.
    """
    below, above = reading["half_below_hz"], reading["half_above_hz"]
    if below is None and above is None:
        return ""
    ends = [
        reading["settled_below_db"] if below is None else None,
        reading["settled_above_db"] if above is None else None,
    ]
    still = max((abs(v) for v in ends if v is not None), default=None)
    return (
        f"  half {'<' if below is None else f'{below:.0f}'}"
        f"-{'>' if above is None else f'{above:.0f}'} Hz"
        + ("" if still is None else f" ({still:.2f} dB left at the end)")
    )


def _steepest_said(reading) -> str:
    """How fast the deviation ran at its fastest, where it has a figure at all."""
    slope = reading.get("steepest_db_per_octave")
    if slope is None:
        return ""
    return f"  {slope:+.1f} dB/oct at {reading['steepest_at_hz']:.0f} Hz"


def cmd_efx_bands(args) -> int:
    """What one parameter did to each band, read from takes already saved."""
    from .. import efxbands

    # Centres given one at a time are read a third of an octave wide, because a
    # list of centres does not say how far each reaches and nothing else in the
    # invocation would say it either.
    named = efxbands.BAND_SETS[args.band_set]

    def said(reading) -> None:
        largest = reading["largest_db"]
        moved = len(reading["outside_the_floor_hz"])
        print(
            f"  {reading['value']:5d} -> "
            + (
                f"{largest:+7.2f} dB at {reading['largest_at_hz']:>6.0f} Hz"
                if largest is not None
                else f"{'inside the floor':>25s}"
            )
            + f"  {moved:2d} bands  {reading['heard_db']:.0f} dBFS"
            + _span_said(reading)
            + _steepest_said(reading)
        )

    found = efxbands.read_directory(
        args.takes,
        type_id=args.type,
        address=args.slot,
        setting=args.setting,
        reference=args.reference,
        control=args.control,
        silence=args.silence,
        stimulus=args.stimulus,
        held=[{"address": a, "bytes": " ".join(f"{b:02X}" for b in v)} for a, v in args.held],
        reference_held=[
            {"address": a, "bytes": " ".join(f"{b:02X}" for b in v)} for a, v in args.reference_held
        ],
        bands_hz=args.band or named[0],
        band_width_octaves=1 / 3 if args.band else named[1],
        channel=args.channel,
        lead_s=args.lead,
        hold_s=args.hold,
        window=tuple(args.window) if args.window else None,
        progress=said,
    )
    picked = found["channel"]
    print(
        f"  read from channel {picked['read']} of "
        f"{len(picked['reference_db'])} ({picked['chosen_by']}): "
        + " ".join(f"{v:.0f}" for v in picked["reference_db"])
        + " dBFS"
    )
    if (beside := found["other_channel"]["read"]) is not None:
        apart = [abs(r["apart_db"]) for r in found["readings"] if r.get("apart_db") is not None]
        moved = [abs(r["other_db"]) for r in found["readings"] if r.get("other_db") is not None]
        if apart:
            print(
                f"  channel {beside} moved up to {max(moved):.2f} dB of its own and "
                f"stayed within {max(apart):.2f} dB of channel {picked['read']}"
            )
    if not found["readings"]:
        print(
            f"no take under {args.takes} has a setting matching {args.setting!r}; "
            f"{found['takes_not_matching']['count']} were looked at"
        )
        return 1
    if not found["control"]["takes"]:
        print("  (no --control: the record cannot say whether the flat setting was unity)")
    if not found["silence"]["takes"]:
        print("  (no --silence: a setting that turns the output off reads as a profile)")
    if missed := found["takes_not_matching"]["count"]:
        print(f"  ({missed} takes under the same directory did not match the pattern)")
    if astray := picked["loudest_elsewhere"]:
        print(
            f"  ({len(astray)} takes are loudest on another channel; read from "
            f"{picked['read']} anyway, and named in the record)"
        )
    report.write_json(args.out, found)
    return 0


def cmd_efx_orders(args) -> int:
    """What one setting did to the orders of a held tone, from takes already saved."""
    from .. import efxorders

    def said(reading) -> None:
        under = reading["under_the_first_db"]
        largest = reading.get("largest_db")
        print(
            f"  {reading['value']:5d} -> "
            + (
                f"{reading['all_of_them_db']:+7.2f} dB under the first"
                if under is not None
                else f"{'not read':>25s}"
            )
            + (
                f"  worst order {reading['largest_at_order']} by {largest:+.2f} dB"
                if largest is not None
                else ""
            )
            + f"  {reading['heard_db']:.0f} dBFS"
        )

    found = efxorders.read_directory(
        args.takes,
        type_id=args.type,
        address=args.slot,
        controller=args.cc,
        setting=args.setting,
        carrier_hz=args.carrier,
        orders=args.orders,
        periods=args.over_periods,
        reference=args.reference,
        control=args.control,
        silence=args.silence,
        stimulus=args.stimulus,
        held=[{"address": a, "bytes": " ".join(f"{b:02X}" for b in v)} for a, v in args.held],
        channel=args.channel,
        lead_s=args.lead,
        hold_s=args.hold,
        progress=said,
    )
    picked = found["channel"]
    print(
        f"  read from channel {picked['read']} of {len(picked['reference_db'])} "
        f"({picked['chosen_by']}, {picked['named_by']}): "
        + " ".join(f"{v:.0f}" for v in picked["reference_db"])
        + " dBFS"
    )
    if not found["readings"]:
        print(
            f"no take under {args.takes} has a setting matching {args.setting!r}; "
            f"{found['takes_not_matching']['count']} were looked at"
        )
        return 1
    if found["reference"]["floor_db"] is None:
        print(
            "  (no --reference: the record carries no floor, so no difference in it "
            "is a reading)"
        )
    else:
        worst = max(found["reference"]["floor_db"])
        print(f"  the repeats of one setting agree to {worst:.2f} dB at the worst order")
    if not found["control"]["takes"]:
        print("  (no --control: an order the carrier arrived with reads as one the stage added)")
    if not found["silence"]["takes"]:
        print("  (no --silence: an order that fell into the floor reads as a harmonic)")
    if missed := found["takes_not_matching"]["count"]:
        print(f"  ({missed} takes under the same directory did not match the patterns)")
    if astray := picked["loudest_elsewhere"]:
        print(
            f"  ({len(astray)} takes are loudest on another channel; read from "
            f"{picked['read']} anyway, and named in the record)"
        )
    report.write_json(args.out, found)
    return 0


def cmd_efx_time(args) -> int:
    """Where a delay slot put its copy, read from takes already saved."""
    from .. import efxtime

    def said(reading) -> None:
        also = "  ".join(
            f"{q:.1f}x{s:.0f}"
            for q, s in zip(reading["also_ms"], reading["also_stands"], strict=True)
        )
        print(
            f"  {reading['value']:5d} -> "
            + (
                f"{reading['ms']:9.3f} ms  x{reading['stands']:6.1f}"
                if reading["admitted"]
                else f"{'under the roughness':>19s}  x{reading['stands']:6.1f}"
            )
            + f"   also {also}"
        )

    found = efxtime.read_directory(
        args.takes,
        type_id=args.type,
        address=args.slot,
        setting=args.setting,
        control=args.control,
        stimulus=args.stimulus,
        held=[{"address": a, "bytes": " ".join(f"{b:02X}" for b in v)} for a, v in args.held],
        channel=args.channel,
        frame=args.frame,
        hop=args.hop,
        searched_ms=(args.shortest, args.longest),
        apart_ms=args.peaks_apart,
        lead_s=args.lead,
        hold_s=args.hold,
        names_a_delay=not args.byte_names_no_delay,
        progress=said,
    )
    picked = found["channel"]
    print(
        f"  read from channel {picked['read']} of "
        f"{len(picked['reference_db'])} ({picked['chosen_by']}): "
        + " ".join(f"{v:.0f}" for v in picked["reference_db"])
        + " dBFS"
    )
    if not found["readings"]:
        print(
            f"no take under {args.takes} has a setting matching {args.setting!r}; "
            f"{found['takes_not_matching']['count']} were looked at"
        )
        return 1
    out = found["with_the_effect_out"]
    print(
        f"  with the effect out the chain's own best peak is {out['ms']:.3f} ms at "
        f"x{out['stands']:.1f}, before anything is taken off it"
    )
    if (null := out["with_nothing_in_its_path"]) is None:
        print("  (only one take with the effect out: this run measured no null)")
    else:
        print(
            f"  put through the same subtraction it reads {null['ms']:.3f} ms at "
            f"x{null['stands']:.1f}"
            + ("  <- which would be read as a delay" if null["admitted"] else "")
        )
    print(f"  searched {found['searched_ms'][0]:.4f} - {found['searched_ms'][1]:.1f} ms")
    left = len(found["settings_asked"]) - len(found["settings_admitted"])
    print(
        f"  {len(found['settings_admitted'])} settings admitted, {left} under the "
        f"roughness, over {found['frames_averaged']} frames a take"
    )
    if found["floor_ms"] is None:
        print("  (no setting was taken twice: this run measured no floor)")
    else:
        print(
            f"  the same setting twice lands {found['floor_ms']:.3f} ms apart, on a "
            f"grid of {found['quefrency_step_ms']:.4f} ms"
        )
    if missed := found["takes_not_matching"]["count"]:
        print(f"  ({missed} takes under the same directory did not match the pattern)")
    if astray := picked["loudest_elsewhere"]:
        print(
            f"  ({len(astray)} takes are loudest on another channel; read from "
            f"{picked['read']} anyway, and named in the record)"
        )
    report.write_json(args.out, found)
    return 0


def cmd_efx_excursion(args) -> int:
    """How far a modulator swings the delay under it, read from takes already saved."""
    from .. import efxexcursion

    def said(reading) -> None:
        swung = (
            f"{reading['excursion_ms']:8.3f} ms"
            if reading["admitted"]
            else f"{(reading['the_level_swing_is'] or 'nothing read'):>11s}"
        )
        span = reading["excursions_that_explain_it_about_as_well_ms"]
        print(
            f"  {reading['value']:5d} -> {swung}"
            + (f"  [{span[0]:.3f} - {span[1]:.3f}]" if span else "")
            + f"   off the phase {reading['off_the_phase_ms']:7.4f} ms"
            + f"   at {reading['read_at_hz']:5.2f} Hz"
        )

    found = efxexcursion.read_directory(
        args.takes,
        type_id=args.type,
        address=args.slot,
        setting=args.setting,
        carrier_hz=args.carrier_hz,
        bypassed=args.bypassed,
        held=[{"address": a, "bytes": " ".join(f"{b:02X}" for b in v)} for a, v in args.held],
        held_not_spelled_out=args.held_not_spelled_out,
        settled_s=args.settled,
        lead_s=args.lead,
        hold_s=args.hold,
        grid_hz=(args.min_rate, args.max_rate),
        step_hz=args.step_hz,
        channel=args.channel,
        progress=said,
    )
    picked = found["channel"]
    print(
        f"  read from channel {picked['read']} of {len(picked['reached_db'])} "
        f"({picked['chosen_by']}): "
        + " ".join(f"{v:.0f}" for v in picked["reached_db"])
        + " dBFS"
    )
    asked = len(found["settings_asked"])
    print(
        f"  {len(found['settings_admitted'])} of {asked} settings named a fitted "
        f"excursion; the phase stood over the bypassed take at "
        f"{len(found['settings_the_phase_admitted'])}"
    )
    if (aside := found["routed_past_the_effect"]["with_nothing_in_its_path"]) is None:
        print("  (only one take with the part routed past: this run measured no null)")
    else:
        print(
            "  with the part routed past the effect, read against another such take, "
            "the same reading returns "
            + (
                f"{aside['excursion_ms']:.3f} ms  <- which would be read as a swing"
                if aside["admitted"]
                else f"{aside['the_level_swing_is'] or 'nothing'}"
            )
        )
    floors = (("fitted", found["floor"]), ("off the phase", found["floor_off_the_phase"]))
    for name, floor in floors:
        if floor["ms"] is None:
            print(f"  ({name}: no setting was taken twice, so this run measured no floor)")
        else:
            print(
                f"  {name}: the same setting {max(floor['takes_per_setting'].values())} times "
                f"spreads {floor['ms']:.4f} ms, {100.0 * floor['as_a_fraction']:.1f} per cent "
                f"of its own mean, over settings {floor['settings_taken_twice']}"
            )
    if missed := found["takes_not_matching"]["count"]:
        print(f"  ({missed} takes under the same directory did not match the pattern)")
    if astray := picked["loudest_elsewhere"]:
        print(
            f"  ({len(astray)} takes are loudest on another channel; read from "
            f"{picked['read']} anyway, and named in the record)"
        )
    report.write_json(args.out, found)
    return 0


def cmd_efx_pair(args) -> int:
    """What one effect did between the unit's two outputs, from takes already saved."""
    from .. import efxbands, efxpair

    def said(reading) -> None:
        lag = (
            f"{reading['lag_us']:+9.2f} us"
            if reading["lag_us"] is not None and reading["one_signal"]
            else f"{'two signals' if not reading['one_signal'] else 'no lag':>12s}"
        )
        moved = reading["within_take_us"]
        print(
            f"  {reading['value']:5d} -> {lag}"
            + (f"   within the take {moved:8.2f} us" if moved is not None else "")
            + f"   apart {reading['apart_db']:+6.1f} dB"
            + f"   {reading['blocks_correlating']}/{reading['blocks']} blocks"
        )

    outputs = None
    if args.outputs_from:
        outputs = json.loads(Path(args.outputs_from).read_text()).get("output_pairs")

    found = efxpair.read_directory(
        args.takes,
        type_id=args.type,
        address=args.slot,
        setting=args.setting,
        bypassed=args.bypassed,
        held=[{"address": a, "bytes": " ".join(f"{b:02X}" for b in v)} for a, v in args.held],
        held_not_spelled_out=args.held_not_spelled_out,
        settled_s=args.settled,
        lead_s=args.lead,
        trim_s=args.trim,
        hold_s=args.hold,
        block=args.block or efxpair.BLOCK,
        bands=efxbands.BAND_SETS[args.band_set][0],
        width_octaves=efxbands.BAND_SETS[args.band_set][1],
        outputs=outputs,
        progress=said,
    )
    picked = found["channel"]
    print(
        f"\n  read across channels {picked['read']} of {len(picked['reached_db'])} "
        f"({picked['chosen_by']}): "
        + " ".join(f"{v:.0f}" for v in picked["reached_db"])
        + " dBFS"
        + ("" if picked["named_by"] else "   (no output map given: the pair is not named)")
    )
    check = found["recovers_an_injected_delay"]
    if check:
        got = check["recovered_us"]
        print(
            f"  a {check['injected_us']:.0f} us delay put in and read back as "
            + (f"{got:+.3f} us (off by {check['off_by_us']:+.3f})" if got is not None
               else "nothing")
            + f"   {check['blocks_correlating']}/{check['blocks']} blocks"
        )
    refused = found["settings_refused"]
    print(
        f"  {len(found['settings_admitted'])} of {len(found['settings_asked'])} "
        "settings gave a lag"
        + (f"; {len(refused)} refused: {refused}" if refused else "")
    )
    floor = found["floor"]
    if floor["across_takes_us"] is None:
        print("  (no setting was taken twice, so this run measured no floor across takes)")
    else:
        print(
            f"  one setting's own takes spread {floor['across_takes_us']:.2f} us, over "
            f"settings {floor['settings_taken_twice']}"
        )
    if floor["widest_within_a_take_us"] is not None:
        print(
            f"  the widest a take disagreed with itself is "
            f"{floor['widest_within_a_take_us']:.2f} us  <- the control"
        )
    if found["routed_past_the_effect"]["with_nothing_in_its_path"] is None:
        print("  (fewer than two takes routed past the effect: this run measured no null)")
    else:
        null = found["routed_past_the_effect"]["with_nothing_in_its_path"]
        print(
            "  with the part routed past the effect, one such take against another, "
            + (f"the lag reads {null['lag_us']:+.2f} us" if null["lag_us"] is not None
               else "no lag is read")
        )
    if missed := found["takes_not_matching"]["count"]:
        print(f"  ({missed} takes under the same directory did not match the pattern)")
    report.write_json(args.out, found)
    return 0


def cmd_efx_params(args) -> int:
    """One verdict per parameter of one type, from records already captured."""
    from .. import efxparams

    types = json.loads(Path(args.types_from).read_text())
    loads = [e["parameters"] for e in types["effects"] if e["type"] == args.type]
    if not loads:
        print(f"{args.types_from} has no type {args.type}")
        return 1
    printed, each, names = None, None, None
    if args.states_from:
        if not args.first_parameter:
            print("--states-from needs --first-parameter to know which block those bytes sit in")
            return 1
        printed = _printed_values(args.states_from, args.type, args.first_parameter)
        each = _printed_settings(args.states_from, printed)
        names = _printed_names(args.states_from, args.type, args.first_parameter)

    found = efxparams.read_directory(
        args.records,
        args.type,
        loads[0],
        args.control,
        [{"address": a, "bytes": " ".join(f"{v:02X}" for v in vs)} for a, vs in args.prepare],
        supersede=dict(s.split("=", 1) for s in args.supersede),
        slots=args.slots or None,
        printed=printed,
        settings=each,
        names=names,
    )
    for name, count in found["results"].items():
        print(f"  {name}: {count}")
    if (coverage := found.get("coverage")) and (missing := coverage["never_asked"]):
        print(f"  => {len(missing)} never asked: {' | '.join(missing)}")
    report.write_json(args.out, found)
    return 0


def cmd_efx_sway(args) -> int:
    """What one modulator does to level and to balance, read from takes already saved."""
    from .. import efxsway

    def said(reading) -> None:
        parts = []
        for key in ("level", "level_in_db", "balance"):
            track = reading[key]
            if track["rate_hz"] is None:
                parts.append(f"{key} --")
                continue
            parts.append(
                f"{key} {track['rate_hz']:.3f} Hz {track['depth']:.3f} "
                f"{track['measured_in']} up {track['going_up_fraction']}"
            )
        print(f"  {reading['value']:5d} -> " + "; ".join(parts))

    found = efxsway.sweep(
        args.takes,
        type_id=args.type,
        address=args.slot,
        setting=args.setting,
        still=args.still,
        held=[{"address": a, "bytes": " ".join(f"{b:02X}" for b in v)} for a, v in args.held],
        lead_s=args.lead,
        search_hz=(args.min_rate, args.max_rate),
        least_db=args.least,
        channels=tuple(args.channels) if args.channels else None,
        progress=said,
    )
    if not found["readings"]:
        print(
            f"no take under {args.takes} has a setting matching {args.setting!r}; "
            f"{found['takes_not_matching']['count']} were looked at"
        )
        return 1
    picked = found["channel"]
    print(
        f"  read from channels {picked['read']} of {len(picked['reached_db'])} "
        f"({picked['chosen_by']}): "
        + " ".join(f"{v:.0f}" for v in picked["reached_db"])
        + " dBFS"
    )
    if found["control"] is not None:
        print(
            f"  control, injected into {found['control_taken_from']}: recovered "
            f"{found['control']['depths_recovered_db']} of "
            f"{found['control']['depths_tried_db']} dB"
        )
    if missed := found["takes_not_matching"]["count"]:
        print(f"  {missed} takes did not match the pattern and were not read")
    report.write_json(args.out, found)
    return 0
