"""Which capture channels carry which socket pair on the back of a unit.

**Two senses of the word, and this reads the coarser one.** A unit with more than one
stereo output has a socket pair per output, and elsewhere in this archive the phrase
`the unit's two outputs` means the left and the right of whichever pair was captured --
the sense a pan byte moves signal between. This reads which capture channels each
socket pair arrived on, and says nothing about which channel of a pair is the left.
Both are needed and neither answers the other; the record names its key `output_pairs`
so that a reader cannot take one for the other.

`takes.channel_pair_reaching` picks the loudest channel and its structural partner,
which is the right rule for finding the pair the unit arrived on -- and it says
nothing about which socket pair that is.

The reading is a routing, not a level. The part is put on one output and struck, then
on the other and struck again, and a channel carrying that output rises when the note
arrives while a channel not carrying it does not. **Levels alone cannot do this here**:
on the takes this was written against, the idle pair sits at -70 dBFS and the pair
carrying the unit reaches -68.8 on the quietest of the six, which is 1.4 dB and inside
what a gain trim decides. The note's arrival is 24 dB clear on the same takes.

What this cannot say is which half of a pair is left and which is right. The part was
struck at one pan position, so both channels of whichever output it was on rose
together. Naming the halves wants the same six takes remade with a pan byte at its two
ends, and that is a booking rather than a reading.

**It is a fact about this rig and this day, not about the unit.** Re-cabling changes
the answer and nothing in a take would say so, which is why the record carries the
capture device by name and why the separation is published rather than asserted.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import takes

MARGIN_DB = 6.0
"""How far a carrying channel's rise must stand over a non-carrying one's.

Not a threshold on the rise itself. The question is whether the two groups separate,
so what matters is the gap between the quietest channel that rose and the loudest that
did not -- and a rise is only evidence about routing if nothing else rose with it.
Six decibels is four times the power and is far under the 24 dB the takes this was
written against actually gave, so the number is a floor on the evidence and not a
tuning.
"""

WHY_THE_NOTE_AND_NOT_THE_LEVEL = (
    "A channel is taken to carry an output because the struck note appears on it, not "
    "because it is loud. An interface has inputs the unit is not plugged into and they "
    "are not silent: on the takes this reading was written against two of them sat at "
    "-70 and -75 dBFS while the pair carrying the unit reached -68.8 on the quietest "
    "take, so a reading by level alone would have turned on 1.4 dB of gain trim. The "
    "note's arrival separates the same takes by 24 dB, because an idle input does not "
    "know when the unit was struck."
)

WHAT_OUTPUT_MEANS = (
    "An output here is a socket pair on the back of the unit, and the key is named "
    "`output_pairs` for that reason. Everywhere else in this archive the phrase `the "
    "unit\'s two outputs` means the two channels of one such pair -- the left and the "
    "right -- which is the sense a pan byte moves signal between and the sense "
    "`takes.other_of_the_pair` returns. The two senses are not the same question and a "
    "record that used one word for both would be read as answering whichever the reader "
    "had in mind. This one answers only which pair is which; which channel of a pair is "
    "the left is in `limits`, unread."
)

WHY_IT_IS_ABOUT_THE_RIG = (
    "This says which socket of the interface each of the unit's outputs was patched "
    "into when these takes were made. Re-cabling changes it and no take would show "
    "that, so every record read on a channel is interpretable only against an output "
    "map made on the same wiring. The capture device is named for that reason."
)


def _rise_per_channel(frames: np.ndarray, rate: int, *, window_s: float) -> list[float]:
    """Each channel in dB over the window after the note, against the window before it.

    The onset is found on the loudest channel of the take rather than per channel,
    because a channel that carries nothing has no onset of its own and would return
    whatever its noise did.
    """
    if frames.ndim == 1:
        frames = frames[:, None]
    envelope = np.abs(frames).max(axis=1)
    floor = float(np.median(envelope))
    over = np.flatnonzero(envelope > 5.0 * floor) if floor > 0 else np.flatnonzero(envelope)
    if over.size == 0:
        raise ValueError("no onset in this take: nothing stands over its own median")
    onset = int(over[0])
    span = int(window_s * rate)
    before, after = frames[max(0, onset - span) : onset], frames[onset : onset + span]
    if before.shape[0] < span // 2 or after.shape[0] < span // 2:
        raise ValueError("the onset sits too close to an end to be read against silence")

    def level(block: np.ndarray) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(np.square(block), axis=0)), 1e-12))

    return [round(float(v), 2) for v in (level(after) - level(before))]


def _control(carried: dict, rows: list[dict], margin_db: float) -> dict:
    """What shows this run could have come out otherwise, as the takes it came out of.

    Each output was struck in more than one state of the chain in front of it. A
    mapping that is a routing gives the same channels in every state; one that is an
    artefact of a state -- a reverb return landing somewhere the dry part does not, an
    effect that moves signal across outputs -- gives different channels in different
    states, and the run would be reading the state rather than the socket.
    """
    per_output = {}
    for name, block in carried.items():
        states = {t["take"]: t["channels"] for t in block["per_take"]}
        apart = [t["apart_db"] for t in block["per_take"] if t["apart_db"] is not None]
        per_output[name] = {
            "states": len(states),
            "channels_each_state_gave": sorted({tuple(v) for v in states.values()}),
            "agreed": len({tuple(v) for v in states.values()}) == 1,
            "narrowest_state_apart_db": round(min(apart), 2) if apart else None,
        }
    return {
        "what": (
            "Each output struck in every state of the chain the run captured, and which "
            "channels rose in each. A routing gives one answer in every state; a reading "
            "of the state rather than of the socket does not."
        ),
        "takes": [row["take"] for row in rows],
        "per_output": per_output,
        "every_state_agreed": all(v["agreed"] for v in per_output.values()),
        "margin_db": margin_db,
    }


def read_directory(
    root: str | Path,
    *,
    output: str,
    window_s: float = 1.0,
    margin_db: float = MARGIN_DB,
    device: str | None = None,
) -> dict:
    """Read one take store and say which channels carry which output.

    `output` is a pattern over each take's setting with a group named `output`, whose
    value is the output the part was put on for that take.
    """
    from scipy.io import wavfile

    root = Path(root)
    manifest = takes.Store.manifest_of(root) if hasattr(takes.Store, "manifest_of") else None
    if manifest is None:
        import json

        manifest = json.loads((root / "takes-manifest.json").read_text())
    pattern = re.compile(output)

    rows, skipped = [], []
    for entry in manifest["takes"]:
        found = pattern.search(entry["setting"])
        if not found:
            skipped.append(entry["setting"])
            continue
        rate, data = wavfile.read(str(root / entry["file"]))
        frames = np.asarray(data, dtype=np.float64)
        if np.issubdtype(np.asarray(data).dtype, np.integer):
            frames = frames / float(np.iinfo(np.asarray(data).dtype).max)
        rows.append({
            "setting": entry["setting"],
            "output": found.group("output"),
            "take": entry["file"],
            "rise_db": _rise_per_channel(frames, rate, window_s=window_s),
            "reached_db": takes.channel_levels(frames),
        })

    if not rows:
        raise ValueError(f"no take under {root} matched {output!r}")

    width = len(rows[0]["rise_db"])
    outputs = sorted({row["output"] for row in rows})
    carried, separations = {}, []
    for name in outputs:
        mine = [row for row in rows if row["output"] == name]
        per_take = []
        for row in mine:
            order = sorted(range(width), key=lambda c: row["rise_db"][c], reverse=True)
            rose = [c for c in order if row["rise_db"][c] >= row["rise_db"][order[0]] - margin_db]
            rest = [c for c in order if c not in rose]
            gap = (
                row["rise_db"][rose[-1]] - row["rise_db"][rest[0]]
                if rest
                else None
            )
            per_take.append({"take": row["take"], "channels": sorted(rose), "apart_db": gap})
            if gap is not None:
                separations.append(gap)
        agreed = {tuple(t["channels"]) for t in per_take}
        carried[name] = {
            "channels": sorted(agreed.pop()) if len(agreed) == 1 else None,
            "every_take_agreed": len(agreed) == 0,
            "per_take": per_take,
        }

    return {
        "question": (
            "which channel of the capture device carries which of the unit's outputs, "
            "read from the arrival of a struck note rather than from a level"
        ),
        "method": {
            "what_is_read": WHY_THE_NOTE_AND_NOT_THE_LEVEL,
            "what_this_is_about": WHY_IT_IS_ABOUT_THE_RIG,
            "window_s": window_s,
            "margin_db": margin_db,
            "capture_device": device,
            "channels": width,
            "takes": len(rows),
        },
        "output_pairs": {
            name: found["channels"] for name, found in carried.items()
        },
        "what_output_means_here": WHAT_OUTPUT_MEANS,
        "how_it_separated": {
            "worst_apart_db": round(min(separations), 2) if separations else None,
            "best_apart_db": round(max(separations), 2) if separations else None,
            "why": (
                "The quietest channel taken to carry an output, against the loudest "
                "taken not to, on the same take. This is the evidence the mapping rests "
                "on, and a run where it narrows toward the margin has stopped "
                "separating the two groups."
            ),
        },
        "each_output": carried,
        "control": _control(carried, rows, margin_db),
        "readings": rows,
        "limits": (
            "Which half of a pair is the left channel and which the right is not read "
            "here. The part was struck at one pan position, so both channels of the "
            "output it was on rose together; the same takes remade with a pan byte at "
            "its two ends would say, and that is a booking rather than a reading. "
            "Nor does this say that any other run was cabled this way: a record read on "
            "a channel is interpretable against an output map made on the same wiring, "
            "and nothing in a take carries the wiring, which is why the capture device "
            "is named. What the reading does say is a routing and not a level, and it "
            "stops being evidence if the two groups stop separating -- `control` is "
            "where that is reported and `how_it_separated` is the margin it is against."
        ),
        "takes_from": str(root),
        "takes_not_matching": {"count": len(skipped), "settings": skipped},
    }


def verdicts(found: dict) -> list[str]:
    """What the reading refuses to conclude, as sentences, or an empty list."""
    wrong = []
    seen: dict[int, str] = {}
    width = found["method"]["channels"]
    for name, block in found["each_output"].items():
        if block["channels"] is None:
            wrong.append(f"output {name}: its takes did not agree on which channels rose")
            continue
        for channel in block["channels"]:
            if channel in seen:
                wrong.append(
                    f"channel {channel} rose for output {seen[channel]} and for {name}"
                )
            seen[channel] = name
        # An output arrives on a pair of the interface's inputs, and the pairing is
        # structural. A found set that is not one says the note was read across two
        # outputs or that a hard-panned take split a pair, and either makes the
        # mapping a level judgement again.
        if len(block["channels"]) == 2:
            first, second = block["channels"]
            if takes.other_of_the_pair(first, width) != second:
                wrong.append(
                    f"output {name} rose on channels {first} and {second}, which the "
                    "interface does not present as a pair"
                )
        elif len(block["channels"]) != 1:
            wrong.append(
                f"output {name} rose on {len(block['channels'])} channels, and an "
                "output arrives on one or two"
            )
    apart = found["how_it_separated"]["worst_apart_db"]
    if apart is not None and apart <= found["method"]["margin_db"]:
        wrong.append(f"the groups separated by {apart} dB, which is inside the margin")
    return wrong
