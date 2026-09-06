"""Asking whether a mapped region ends where the map says it ends.

A region's size is what a block read of that region returned. Twice it has turned
out not to be the extent of the addresses behind it: a block the map bounds at
one size answered single-byte reads above it, and one of the addresses found that
way was the byte that moves a part to the second output pair.

That matters past the two blocks it was seen in. The write probe, the hold probe
and every block plan are built from those sizes, so a region reported short is a
run of addresses that no stage has asked anything of -- not because they were
judged uninteresting but because nothing knew they were there.

**A null here is narrow.** An address that answers nothing to a single-byte read
is not thereby an address that does not exist: this unit has a run of them whose
contents come back in a block read starting earlier. So a region this reports as
ending where the map says is bounded by what a single-byte read can reach, and no
further. The two failures are opposite and neither covers the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: What the stage is bounded by, carried into the record because the whole result
#: is a list of negatives and a negative is worth reading only with its bound.
LIMIT = (
    "Every question here is a single-byte read. An address that answers one is there; an "
    "address that answers nothing may still be reached by a block read that starts earlier, "
    "which this unit is known to do. So a region reported as ending where the map says is "
    "bounded by what a single-byte read reaches."
)

#: Why the run carries a canary.
WHY_THE_CANARY = (
    "An address known to answer, asked between regions. A unit that stops talking answers "
    "nothing to every question after that, which is exactly what a region ending where the "
    "map says looks like. Without this, a run that lost the unit early reports a whole map "
    "of regions confirmed."
)

WENT_DEAF = "the canary stopped answering, so nothing after the last one it answered was measured"


@dataclass
class Region:
    """One region, and how far past its mapped end anything answered."""

    address: str
    mapped_size: int
    answered_beyond: int = 0
    stopped_at: str | None = None
    values: list[str] = field(default_factory=list)

    @property
    def short(self) -> bool:
        return self.answered_beyond > 0

    def to_json(self) -> dict:
        out = {
            "address": self.address,
            "mapped_size": self.mapped_size,
            "answered_beyond_the_mapped_end": self.answered_beyond,
        }
        if self.answered_beyond:
            out["first_address_past_the_end"] = self.stopped_at
            out["values"] = self.values
        return out


def restore(row: dict) -> Region:
    """A region read back from a record, so an interrupted run resumes rather than repeats."""
    return Region(
        address=row["address"],
        mapped_size=row["mapped_size"],
        answered_beyond=row["answered_beyond_the_mapped_end"],
        stopped_at=row.get("first_address_past_the_end"),
        values=list(row.get("values", [])),
    )


def past_the_end(address: str, size: int) -> int:
    """The packed address one byte past a region, as three seven-bit bytes hold it."""
    high, mid, low = (int(part, 16) for part in address.split())
    return (high << 14 | mid << 7 | low) + size


def unpack(packed: int) -> str:
    return " ".join(f"{(packed >> shift) & 0x7F:02X}" for shift in (14, 7, 0))


def summarise(regions: list[Region], canary: str, deaf: bool) -> dict:
    """The regions that ran past their mapped end, and the bound on the rest."""
    short = [r for r in regions if r.short]
    return {
        "asked": len(regions),
        "regions_reaching_past_their_mapped_end": len(short),
        "addresses_found_that_way": sum(r.answered_beyond for r in short),
        "note": LIMIT,
        "positive_control": {"canary": canary, "why": WHY_THE_CANARY},
        "stopped": WENT_DEAF if deaf else None,
        "regions": [r.to_json() for r in regions],
    }
