"""Map an address space by asking the machine, hierarchically.

Three properties of the device make an exhaustive probe unnecessary.

An oversized read is answered with whatever the region actually holds, so one
round trip can bound a region rather than one round trip per byte. But that is
not uniform: some regions refuse an oversized request outright and return
nothing, and a silent reply therefore does not mean the region is absent. Sizes
are bisected rather than assumed.

An address that does not exist stays silent, so the cost of the sweep is set by
the timeout on misses rather than by the round trip on hits. The timeout is kept
just above the observed worst case, and `selftest` is what establishes what that
worst case is.

And a silent address is indistinguishable from a silent machine. A unit that
stops answering part way through a sweep turns every remaining address into a
false absence, and the run still completes and still writes a map -- one that
looks like a sparse address space rather than a broken measurement. So the sweep
re-checks a known-good address at intervals and aborts the moment it stops
answering, rather than carrying on and recording the silence as data.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from . import roland
from .midi import MidiLink

# Probed at each top byte to decide whether it leads anywhere. A region need not
# begin at 00 00, so a single probe there would miss a block that starts higher.
LEVEL1_PROBES = (0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x40)


@dataclass
class Region:
    address: str
    size: int
    """Bytes the device returned for a bounded request. 0 means it never answered."""

    data: list[int] = field(default_factory=list)
    oversize_behaviour: str = ""
    """'truncates' if an oversized request came back short, 'refuses' if it came back empty."""

    checksum_ok: bool = True


@dataclass
class SweepResult:
    device_id: int
    timeout: float
    probes_sent: int = 0
    seconds: float = 0.0
    top_bytes: list[int] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    aborted: str = ""
    """Why the sweep stopped early. A non-empty value means the map is incomplete."""

    @property
    def complete(self) -> bool:
        return not self.aborted

    def to_json(self) -> dict:
        return {
            "device_id": f"{self.device_id:02X}",
            "complete": self.complete,
            "aborted": self.aborted or None,
            "probe_timeout_s": self.timeout,
            "probes_sent": self.probes_sent,
            "elapsed_s": round(self.seconds, 1),
            "top_bytes_answering": [f"{b:02X}" for b in self.top_bytes],
            "regions": [
                {
                    "address": r.address,
                    "size": r.size,
                    "data": " ".join(f"{b:02X}" for b in r.data),
                    "oversize_behaviour": r.oversize_behaviour,
                    "checksum_ok": r.checksum_ok,
                }
                for r in self.regions
            ],
        }


class DeviceWentSilent(RuntimeError):
    """The canary address stopped answering, so the run is measuring a dead machine."""


class Sweeper:
    def __init__(
        self,
        link: MidiLink,
        *,
        device_id: int = roland.DEFAULT_DEVICE_ID,
        timeout: float = 0.06,
        canary: tuple[int, int, int] = (0x40, 0x01, 0x30),
        canary_every: int = 64,
        ceiling: int = 64,
    ):
        self.link = link
        self.device_id = device_id
        self.timeout = timeout
        self.canary = canary
        self.canary_every = canary_every
        self.ceiling = ceiling
        self.probes = 0
        self._since_canary = 0

    def read(self, address: tuple[int, int, int], size: int) -> roland.Dt1Reply | None:
        self.probes += 1
        request = roland.rq1(address, size, device_id=self.device_id)
        return roland.parse_dt1(self.link.exchange(request, timeout=self.timeout))

    def check_alive(self) -> None:
        """Confirm a known-good address still answers, generously.

        Uses a long timeout on purpose: the question here is whether the machine
        is alive at all, not whether it is quick, and a canary that fails on a
        merely slow reply would abort good runs.
        """
        self.probes += 1
        request = roland.rq1(self.canary, 1, device_id=self.device_id)
        if roland.parse_dt1(self.link.exchange(request, timeout=1.0)) is None:
            addr = " ".join(f"{b:02X}" for b in self.canary)
            raise DeviceWentSilent(
                f"the canary address {addr} stopped answering after {self.probes} probes; "
                "everything measured past this point would be a false absence"
            )
        self._since_canary = 0

    def answers(self, address: tuple[int, int, int]) -> bool:
        got = self.read(address, 1) is not None
        self._since_canary += 1
        # Only a miss is ambiguous, so only a run of misses needs the canary.
        if not got and self._since_canary >= self.canary_every:
            self.check_alive()
        return got

    def bound_region(self, address: tuple[int, int, int], *, ceiling: int | None = None) -> Region:
        """Find how many bytes the device will hand over starting at this address.

        An oversized request either truncates at the region boundary or is
        refused. The first case answers in one probe; the second needs a bisection
        between a size known to work and one known not to.

        The ceiling is kept modest. A request far larger than any real region is
        not more informative -- it only asks the machine to do something no file
        would ever ask of it, and this unit stopped responding entirely during a
        sweep that used 512.
        """
        ceiling = self.ceiling if ceiling is None else ceiling
        addr = f"{address[0]:02X} {address[1]:02X} {address[2]:02X}"
        big = self.read(address, ceiling)
        if big is not None and big.size < ceiling:
            return Region(addr, big.size, big.data, "truncates", big.checksum_ok)
        if big is not None:
            return Region(addr, big.size, big.data, "at least the ceiling", big.checksum_ok)

        lo, hi = 1, ceiling  # lo answers, hi does not
        first = self.read(address, lo)
        if first is None:
            return Region(addr, 0, [], "no reply at any size")
        best = first
        while hi - lo > 1:
            mid = (lo + hi) // 2
            reply = self.read(address, mid)
            if reply is None:
                hi = mid
            else:
                lo, best = mid, reply
        return Region(addr, best.size, best.data, "refuses", best.checksum_ok)

    def run(
        self, *, top_bytes: range | None = None, low_bytes: tuple[int, ...] = (0x00,), progress=None
    ) -> SweepResult:
        """Sweep, and say plainly if it did not finish.

        `low_bytes` is the set of third bytes tried under each (top, mid) pair. A
        region need not begin at 00 -- the system effect block on the unit this
        was written against begins at 30 -- so a sweep that only ever probes 00
        will bound the first region of a pair and never see the rest.
        """
        started = time.monotonic()
        result = SweepResult(device_id=self.device_id, timeout=self.timeout)
        try:
            self.check_alive()
            for top in top_bytes if top_bytes is not None else range(0x80):
                if any(self.answers((top, mid, 0x00)) for mid in LEVEL1_PROBES):
                    result.top_bytes.append(top)
                    if progress:
                        progress(f"top byte {top:02X}: answers")
            if progress:
                progress(
                    f"level 1 done: {len(result.top_bytes)} top bytes answer ({self.probes} probes)"
                )

            seen: set[tuple[int, int, int]] = set()
            for top in result.top_bytes:
                found = 0
                for mid in range(0x80):
                    for low in low_bytes:
                        addr = (top, mid, low)
                        if addr in seen or not self.answers(addr):
                            continue
                        region = self.bound_region(addr)
                        result.regions.append(region)
                        # A truncating region tells us where the next one starts.
                        if region.oversize_behaviour == "truncates" and region.size:
                            for covered in range(low, min(low + region.size, 0x80)):
                                seen.add((top, mid, covered))
                        found += 1
                if progress:
                    progress(f"top byte {top:02X}: {found} regions")
        except DeviceWentSilent as exc:
            result.aborted = str(exc)
            if progress:
                progress(f"ABORTED: {exc}")

        result.probes_sent = self.probes
        result.seconds = time.monotonic() - started
        return result


def summarise(result: SweepResult) -> str:
    lines = []
    if not result.complete:
        lines.append(f"INCOMPLETE -- {result.aborted}")
        lines.append("The regions below are what was measured before it stopped, not a map.")
    lines.append(
        f"probes {result.probes_sent}, {result.seconds:.1f}s, "
        f"{len(result.regions)} regions across {len(result.top_bytes)} top bytes"
    )
    by_behaviour: dict[str, int] = {}
    for r in result.regions:
        by_behaviour[r.oversize_behaviour] = by_behaviour.get(r.oversize_behaviour, 0) + 1
    for k, v in sorted(by_behaviour.items()):
        lines.append(f"  {v:4d} regions {k}")
    bad = [r for r in result.regions if not r.checksum_ok]
    if bad:
        lines.append(f"  !! {len(bad)} regions returned a bad checksum -- the path corrupted them")
    total = sum(r.size for r in result.regions)
    lines.append(f"  {total} bytes readable in total")
    return "\n".join(lines)


__all__ = ["Region", "SweepResult", "Sweeper", "summarise", "asdict"]
