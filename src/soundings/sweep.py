"""Map an address space by asking the machine, hierarchically.

Five properties of the device shape the probe.

An oversized read is answered with whatever the region actually holds, so one
round trip can bound a region rather than one round trip per byte. But that is
not uniform: some regions refuse an oversized request outright and return
nothing, and a silent reply therefore does not mean the region is absent. Sizes
are bisected rather than assumed.

An address that does not exist stays silent, so the cost of the sweep is set by
the deadline on misses rather than by the round trip on hits.

**A deadline is a measurement, and a badly measured one corrupts everything
after it.** Reply time grows with reply length -- one byte in 10 ms, sixty four
in 78 -- so a single constant cannot serve both ends. But length is not the
whole story either: two addresses on this unit both return 64 bytes, one in 31
ms and one in 78. A deadline is only safe if it bounds the slowest address, not
the one it happened to be measured on.

Getting it wrong does not merely lose the slow replies. A late reply is not
discarded; it arrives while the next probe is listening, and from then on every
reply is one request behind. Pairing on the address does not catch that, because
a size bisection asks the same address every time. A run with a flat 60 ms
deadline reported 655 regions as returning a full 64 bytes when each had
actually read the previous request's answer, and a run whose calibration address
refused oversized reads fitted a slope three times too shallow from the short
end alone and did the same thing. So the fit is measured across several
addresses and required to span the sizes it will be used at, and a reply longer
than what was asked for is treated as proof the deadline is wrong rather than as
data.

**A read request is not always a read.** Some addresses are commands: the unit
this was written against answers a one-byte request at 0C 00 00 with a 286
message, 29 second bulk dump of an entirely different part of the address space.
Every probe sent during that dump takes a dump message as its own answer, so one
such address makes the rest of the sweep report whatever it happens to be
transmitting. That is not a rare corner -- it is what made a first run of this
sweep report 117 of 128 top bytes as populated. So a reply is paired against the
address that was asked for, an unpaired reply is treated as a stream rather than
as an answer, and the sweep waits for the machine to stop talking before it
probes again.

And a silent address is indistinguishable from a silent machine. A unit that
stops answering part way through turns every remaining address into a false
absence, and the run still completes and still writes a map -- one that looks
like a sparse address space rather than a broken measurement. So the sweep
re-checks a known-good address at intervals. Silence there aborts the run;
a reply for some other address does not, because a machine talking about
something else is a machine that is alive.
"""

from __future__ import annotations

import enum
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
class Stream:
    """A transmission the machine started on its own, and where the sweep noticed it.

    Recorded rather than discarded: an address that dumps is a finding, and the
    sizes and addresses it emits say what it dumped.

    `first_seen_after` is not the trigger. A dump whose first message arrives
    after its own probe's deadline is noticed by the probe after that one, so the
    field names where the sweep was standing, not what it stood on: the same dump
    was attributed to 0C 00 00 under one deadline and 0C 01 00 under another.
    Naming the trigger takes the isolation procedure -- probe one address, then
    listen in silence -- which is not something a sweep can do at every step.
    """

    first_seen_after: str
    messages: int
    seconds: float
    payload_sizes: list[int] = field(default_factory=list)
    addresses: list[str] = field(default_factory=list)
    settled: bool = True
    """False if the machine was still transmitting when the drain gave up."""


@dataclass
class SweepResult:
    device_id: int
    model_id: int
    timing: Timing
    probes_sent: int = 0
    lagged: int = 0
    """Replies that outran their deadline. Non-zero condemns the whole run."""

    lag_lengths: dict[int, int] = field(default_factory=dict)
    malformed: int = 0
    """Frames that were not legal SysEx. Recovered by a retry, but a fault worth counting."""

    malformed_reasons: dict[str, int] = field(default_factory=dict)

    seconds: float = 0.0
    top_bytes: list[int] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    streams: list[Stream] = field(default_factory=list)
    aborted: str = ""
    """Why the sweep stopped early. A non-empty value means the map is incomplete."""

    @property
    def complete(self) -> bool:
        return not self.aborted

    def to_json(self) -> dict:
        return {
            "device_id": f"{self.device_id:02X}",
            "model_id": f"{self.model_id:02X}",
            "complete": self.complete,
            "aborted": self.aborted or None,
            "trustworthy": self.complete and self.lagged == 0,
            "reply_timing": {
                "base_ms": round(self.timing.base_ms, 2),
                "per_byte_ms": round(self.timing.per_byte_ms, 3),
                "margin": self.timing.margin,
                "timeout_ms_at_size_1": round(self.timing.timeout(1) * 1000, 1),
                "timeout_ms_at_size_64": round(self.timing.timeout(64) * 1000, 1),
                "fitted_over_reply_lengths": self.timing.fitted_over,
            },
            "replies_lagged": self.lagged,
            "malformed_frames": self.malformed,
            "malformed_frame_reasons": self.malformed_reasons,
            "lagged_reply_lengths": {str(k): v for k, v in sorted(self.lag_lengths.items())},
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
            "streams": [
                {
                    "first_seen_after": s.first_seen_after,
                    "messages": s.messages,
                    "seconds": round(s.seconds, 1),
                    "payload_sizes": s.payload_sizes,
                    "addresses": s.addresses,
                    "settled": s.settled,
                }
                for s in self.streams
            ],
        }


class DeviceWentSilent(RuntimeError):
    """The canary address said nothing at all, so the run is measuring a dead machine."""


class MidiCalibrationError(RuntimeError):
    """The unit did not give enough distinct reply lengths to fit a deadline against."""


class Probe(enum.Enum):
    HIT = "hit"
    MISS = "miss"
    STREAM = "stream"
    """The machine replied, but about a different address. It is transmitting, not answering."""

    LAGGED = "lagged"
    """A reply carrying more bytes than were asked for, so it belongs to an earlier request."""

    MALFORMED = "malformed"
    """The frame that came back was not one legal SysEx message, twice running."""


@dataclass
class Timing:
    """How long a reply takes, as a function of how much was asked for.

    A single timeout cannot serve a probe that asks for one byte and a probe that
    asks for sixty four: the reply is sent down a serial line, so its arrival is
    dominated by its own length. Calibrating on a small read and applying the
    number to a large one makes every large read time out -- and a timed out
    reply is not lost, it arrives while the next probe is listening. From there
    every reply is one request behind, and the lag is invisible whenever
    consecutive probes share an address, which inside a bisection they all do.

    So the two coefficients are measured on the unit rather than chosen, and
    `margin` is what separates a slow reply from an absent one.
    """

    base_ms: float
    per_byte_ms: float
    margin: float = 3.0
    floor_ms: float = 10.0
    fitted_over: list[int] = field(default_factory=list)
    """Reply lengths the fit was measured at. A deadline outside this range is a guess."""

    def timeout(self, size: int) -> float:
        return (self.floor_ms + (self.base_ms + self.per_byte_ms * size) * self.margin) / 1000.0

    def describe(self) -> str:
        span = f", fitted over {self.fitted_over} byte replies" if self.fitted_over else ""
        return (
            f"{self.base_ms:.1f} ms + {self.per_byte_ms:.2f} ms/byte measured, "
            f"x{self.margin:g} margin => {self.timeout(1) * 1000:.0f} ms at size 1, "
            f"{self.timeout(64) * 1000:.0f} ms at size 64{span}"
        )


def calibrate(
    link: MidiLink,
    *,
    addresses: list[tuple[int, int, int]],
    device_id: int = roland.DEFAULT_DEVICE_ID,
    model_id: int = roland.GS_MODEL_ID,
    sizes: tuple[int, ...] = (1, 16, 64),
    repeats: int = 6,
    require_length: int = 0,
    probe_timeout: float = 1.0,
) -> Timing:
    """Measure reply latency against reply length on this unit.

    Two things this does that a single well chosen address cannot.

    It reads several addresses, because latency is not a property of length
    alone: on the unit this was written against, two addresses both returning 64
    bytes answer in 31 ms and 78 ms. A deadline is only safe if it bounds the
    slowest, so the worst time seen at each reply length is what gets fitted.

    And it insists the fit spans the sizes it will be used over. An address that
    refuses oversized reads yields no long point at all, and the fit silently
    becomes a short range extrapolation -- one canary that refused a 64 byte read
    produced a slope three times too shallow, and every large read in the sweep
    that followed missed its deadline. `require_length` turns that into an error
    instead of a quiet miscalibration.

    Fitted against the bytes the machine returned rather than the bytes asked
    for, so a truncating address still gives a usable point. The worst of each
    repeat is used, not the median: a deadline set on typical behaviour is missed
    routinely.
    """
    worst_at: dict[int, float] = {}
    for address in addresses:
        for size in sizes:
            # The first probe decides whether this pair is worth repeating. Most
            # addresses refuse most sizes, and every refusal costs the full
            # timeout; without this, a calibration over a hundred addresses is
            # slower than the sweep it exists to make possible.
            for _ in range(repeats):
                link.drain()
                started = time.monotonic()
                link.send(roland.rq1(address, size, device_id=device_id, model_id=model_id))
                reply = roland.parse_dt1(link.receive(timeout=probe_timeout))
                if reply is None:
                    link.drain()
                    break
                if reply.address != address or reply.size > size:
                    # Not an answer to this request. Some addresses are commands
                    # that start a dump, and its messages arrive fast and long --
                    # timed as replies they flatten the fit to nothing, which is
                    # the very failure this function exists to prevent. Wait the
                    # transmission out and take no point from it.
                    while link.receive(timeout=1.5):
                        pass
                    break
                elapsed = (time.monotonic() - started) * 1000.0
                worst_at[reply.size] = max(worst_at.get(reply.size, 0.0), elapsed)
    points = sorted(worst_at.items())
    if len(points) < 2:
        raise MidiCalibrationError(
            f"calibration needs replies of two different lengths; got {[n for n, _ in points]} "
            f"from {len(addresses)} addresses"
        )
    if require_length and points[-1][0] < require_length:
        raise MidiCalibrationError(
            f"calibration reached {points[-1][0]} bytes but the sweep reads up to "
            f"{require_length}; fitting past that is extrapolation, and a deadline "
            "extrapolated short makes every large read arrive after the next request"
        )
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    var = sum((x - mean_x) ** 2 for x, _ in points)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in points)
    # The slope comes from least squares, but the intercept is raised until the
    # line sits above every point. A deadline that does not bound its own
    # measurements is not a deadline, and a least squares line leaves half of
    # them above it by construction.
    per_byte = max(0.0, cov / var)
    return Timing(
        base_ms=max(y - per_byte * x for x, y in points),
        per_byte_ms=per_byte,
        fitted_over=[n for n, _ in points],
    )


class Sweeper:
    def __init__(
        self,
        link: MidiLink,
        *,
        timing: Timing,
        device_id: int = roland.DEFAULT_DEVICE_ID,
        model_id: int = roland.GS_MODEL_ID,
        canary: tuple[int, int, int] = (0x40, 0x01, 0x30),
        canary_every: int = 64,
        ceiling: int = 64,
    ):
        self.link = link
        self.device_id = device_id
        self.model_id = model_id
        self.timing = timing
        self.canary = canary
        self.canary_every = canary_every
        self.ceiling = ceiling
        self.probes = 0
        self.lagged = 0
        """Replies that outran their own deadline. Any at all means the calibration is wrong."""

        self.lag_lengths: dict[int, int] = {}
        """How many lagged replies were of each length, which names the deadline that failed."""

        self.malformed = 0
        self.malformed_reasons: dict[str, int] = {}
        """Frames that were not legal SysEx, by reason. These are a path fault, not a reading."""

        self.streams: list[Stream] = []
        self._since_canary = 0

    def drain_to_quiet(self, *, settle: float = 1.5, cap: float = 90.0) -> Stream | None:
        """Wait out a transmission, recording what it was.

        Returns None if the machine was already quiet. The cap exists because a
        dump of unknown length must not be able to stall the sweep indefinitely;
        hitting it is reported rather than hidden, since a partially drained
        stream leaves the next probe reading dump traffic again.
        """
        started = time.monotonic()
        sizes: set[int] = set()
        addresses: list[str] = []
        count = 0
        while time.monotonic() - started < cap:
            raw = self.link.receive(timeout=settle)
            if not raw:
                break
            count += 1
            reply = roland.parse_dt1(raw)
            if reply is not None:
                sizes.add(reply.size)
                addresses.append(" ".join(f"{b:02X}" for b in reply.address))
        if count == 0:
            return None
        return Stream(
            first_seen_after="",
            messages=count,
            seconds=time.monotonic() - started - settle,
            payload_sizes=sorted(sizes),
            addresses=_span(addresses),
            settled=time.monotonic() - started < cap,
        )

    def read(
        self, address: tuple[int, int, int], size: int
    ) -> tuple[Probe, roland.Dt1Reply | None]:
        """Read one address, and check the reply is an answer to this request.

        Two ways it can fail to be. A reply for a different address means the
        machine is transmitting something of its own, and until it stops, every
        further probe reads that transmission instead of an answer -- so the
        stream is recorded against the address that started it and waited out in
        full. A reply for this address but longer than was asked for belongs to
        an earlier request that outran its deadline, which the address alone
        cannot reveal because a bisection asks the same address every time.

        The second check is not exhaustive: a lagged reply shorter than the
        current request passes it. It does not need to be exhaustive. Lag is a
        run-wide condition, not a per-probe accident, so catching any of it is
        enough to condemn the calibration that caused it.
        """
        for attempt in (0, 1):
            self.probes += 1
            request = roland.rq1(address, size, device_id=self.device_id, model_id=self.model_id)
            raw = self.link.exchange(request, timeout=self.timing.timeout(size))
            broken = roland.malformation(raw)
            if broken is None:
                break
            # The frame is not a message, so its length is not a length. Wait for
            # the path to go quiet and ask once more; a second failure is left as
            # a miss rather than guessed at.
            self.malformed += 1
            self.malformed_reasons[broken] = self.malformed_reasons.get(broken, 0) + 1
            self.drain_to_quiet(settle=0.5)
            if attempt:
                return Probe.MALFORMED, None

        reply = roland.parse_dt1(raw)
        if reply is None:
            return Probe.MISS, None
        # The model id is part of what makes the address this address. A reply
        # carrying the same three bytes under another one answers a different
        # question, and counted as a hit it would put a region on the map of a
        # space that was never asked.
        mine = reply.address == address and reply.model_id == self.model_id
        if mine and reply.size <= size:
            return Probe.HIT, reply

        addr = f"{address[0]:02X} {address[1]:02X} {address[2]:02X}"
        stale = mine
        if stale:
            # The reply's own length says which request outran its deadline, and
            # that is the only thing that says which deadline was wrong.
            self.lagged += 1
            self.lag_lengths[reply.size] = self.lag_lengths.get(reply.size, 0) + 1
        stream = self.drain_to_quiet() or Stream("", 0, 0.0)
        if not stale:
            stream.first_seen_after = addr
            stream.messages += 1
            stream.addresses = _span(
                [" ".join(f"{b:02X}" for b in reply.address), *stream.addresses]
            )
            stream.payload_sizes = sorted({reply.size, *stream.payload_sizes})
            self.streams.append(stream)
        return (Probe.LAGGED if stale else Probe.STREAM), None

    def check_alive(self) -> None:
        """Confirm a known-good address still answers.

        Only silence counts as death, and only after three asks. Everything else
        that can come back is evidence of a live machine and has at some point
        been mistaken for a dead one here: a merely slow reply, a reply for a
        different address while a dump is running, and a frame too malformed to
        parse. The first version of this check aborted a run on a unit that
        answered an identity request the moment the run gave up; the second
        aborted on its very first probe because a malformed frame parsed to None
        and None was read as nothing.
        """
        request = roland.rq1(self.canary, 1, device_id=self.device_id, model_id=self.model_id)
        for _ in range(3):
            self.drain_to_quiet(settle=0.5)
            self.probes += 1
            raw = self.link.exchange(request, timeout=1.0)
            if raw:
                # Anything at all came back. It may be a malformed frame or a
                # reply for another address, and neither is an answer -- but both
                # are a machine that is transmitting, which is the only question
                # being asked here.
                self._since_canary = 0
                return
        addr = " ".join(f"{b:02X}" for b in self.canary)
        raise DeviceWentSilent(
            f"the canary address {addr} said nothing to three requests after "
            f"{self.probes} probes; everything measured past this point would be a false absence"
        )

    def answers(self, address: tuple[int, int, int]) -> bool:
        outcome, _ = self.read(address, 1)
        self._since_canary += 1
        # Only a miss is ambiguous, so only a run of misses needs the canary.
        if outcome is Probe.MISS and self._since_canary >= self.canary_every:
            self.check_alive()
        return outcome is Probe.HIT

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
        outcome, big = self.read(address, ceiling)
        if outcome is Probe.STREAM:
            return Region(addr, 0, [], "dumps")
        if big is not None and big.size < ceiling:
            return Region(addr, big.size, big.data, "truncates", big.checksum_ok)
        if big is not None:
            return Region(addr, big.size, big.data, "at least the ceiling", big.checksum_ok)

        lo, hi = 1, ceiling  # lo answers, hi does not
        outcome, first = self.read(address, lo)
        if first is None:
            return Region(
                addr, 0, [], "dumps" if outcome is Probe.STREAM else "no reply at any size"
            )
        best = first
        while hi - lo > 1:
            mid = (lo + hi) // 2
            outcome, reply = self.read(address, mid)
            if reply is None:
                hi = mid
            else:
                lo, best = mid, reply
        return Region(addr, best.size, best.data, "refuses", best.checksum_ok)

    def recalibrate(
        self, addresses: list[tuple[int, int, int]], *, sample: int = 24, progress=None
    ) -> None:
        """Re-fit the deadline over every address known to answer, before reading in bulk.

        All of them, not the first that reaches the ceiling. Stopping early is
        what a cheap version of this did, and on this unit it happened to land on
        a fast address: the fit came out at 0.34 ms per byte with a threefold
        margin, which is only a 1.3-fold margin over the slowest address actually
        present. The margin has to be a margin over the worst case, and the worst
        case is not visible from one address.

        The list is sampled rather than exhausted. Most addresses refuse most
        sizes and every refusal costs a full timeout, so calibrating over all of
        them takes longer than the sweep does; the sample is spread across the
        list so it does not consist of one top byte's neighbours.

        Failing to reach the ceiling at all is reported and does not stop the
        run. The sweep then knows its deadline is an extrapolation, which is
        worth saying out loud rather than presenting as measured.
        """
        stride = max(1, len(addresses) // sample)
        chosen = addresses[::stride][:sample]
        try:
            timing = calibrate(
                self.link,
                addresses=chosen,
                device_id=self.device_id,
                model_id=self.model_id,
                sizes=(1, self.ceiling // 4, self.ceiling),
                require_length=self.ceiling,
                probe_timeout=0.5,
            )
        except MidiCalibrationError as exc:
            if progress:
                progress(f"deadline stays extrapolated ({exc}): {self.timing.describe()}")
            return
        timing.margin = self.timing.margin
        self.timing = timing
        if progress:
            progress(
                f"timing re-measured over {len(chosen)} of {len(addresses)} addresses: "
                f"{timing.describe()}"
            )

    def run(
        self, *, top_bytes: range | None = None, low_bytes: tuple[int, ...] = (0x00,), progress=None
    ) -> SweepResult:
        """Sweep, and say plainly if it did not finish.

        `low_bytes` is the set of third bytes tried under each (top, mid) pair. A
        region need not begin at 00 -- the system effect block on the unit this
        was written against begins at 30 -- so a sweep that only ever probes 00
        will bound the first region of a pair and never see the rest.

        Timing is calibrated twice, because the two levels ask for different
        amounts. Level 1 only ever reads one byte, so a one byte deadline is all
        it needs and all that can honestly be measured before any address is
        known. Level 2 reads up to the ceiling, and the addresses level 1 found
        are what makes a deadline for that measurable rather than extrapolated.
        """
        started = time.monotonic()
        result = SweepResult(device_id=self.device_id, model_id=self.model_id, timing=self.timing)
        try:
            self.check_alive()
            # Every probe is tried rather than stopping at the first hit per top
            # byte. The extra probes are a rounding error against the run, and
            # the addresses they find are what the bulk deadline is measured on
            # -- a candidate list of (top, 00, 00) alone would have missed the
            # slowest address on this unit, which sits at 40 02 00.
            hits: list[tuple[int, int, int]] = []
            for top in top_bytes if top_bytes is not None else range(0x80):
                found = [mid for mid in LEVEL1_PROBES if self.answers((top, mid, 0x00))]
                if found:
                    result.top_bytes.append(top)
                    hits.extend((top, mid, 0x00) for mid in found)
                    if progress:
                        progress(f"top byte {top:02X}: answers")
            if progress:
                progress(
                    f"level 1 done: {len(result.top_bytes)} top bytes answer, "
                    f"{len(hits)} addresses ({self.probes} probes)"
                )

            self.recalibrate(hits, progress=progress)
            result.timing = self.timing

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
        result.lagged = self.lagged
        result.lag_lengths = self.lag_lengths
        result.malformed = self.malformed
        result.malformed_reasons = self.malformed_reasons
        result.streams = self.streams
        result.seconds = time.monotonic() - started
        return result


def _span(addresses: list[str]) -> list[str]:
    """Keep the ends of a long address list rather than all of it."""
    if len(addresses) <= 8:
        return addresses
    return [*addresses[:4], f"... {len(addresses) - 8} more ...", *addresses[-4:]]


def summarise(result: SweepResult) -> str:
    lines = []
    if not result.complete:
        lines.append(f"INCOMPLETE -- {result.aborted}")
        lines.append("The regions below are what was measured before it stopped, not a map.")
    if result.lagged:
        lines.append(
            f"UNTRUSTWORTHY -- {result.lagged} replies outran their deadline, so replies were "
            "running behind requests and the sizes below are not the sizes asked for."
        )
        lines.append(f"The timing was {result.timing.describe()}; it is too tight.")
        for length, count in sorted(result.lag_lengths.items()):
            lines.append(
                f"  {count} of them were {length} byte replies, whose deadline was "
                f"{result.timing.timeout(length) * 1000:.0f} ms"
            )
    lines.append(
        f"probes {result.probes_sent}, {result.seconds:.1f}s, "
        f"{len(result.regions)} regions across {len(result.top_bytes)} top bytes"
    )
    by_behaviour: dict[str, int] = {}
    for r in result.regions:
        by_behaviour[r.oversize_behaviour] = by_behaviour.get(r.oversize_behaviour, 0) + 1
    for k, v in sorted(by_behaviour.items()):
        lines.append(f"  {v:4d} regions {k}")
    for s in result.streams:
        note = "" if s.settled else "  (STILL TRANSMITTING when the drain gave up)"
        lines.append(
            f"  a stream appeared at or before {s.first_seen_after}: "
            f"{s.messages} messages over {s.seconds:.1f}s, "
            f"payload sizes {s.payload_sizes}{note}"
        )
    if result.malformed:
        lines.append(
            f"  {result.malformed} frames came back that were not legal SysEx and were re-asked:"
        )
        for reason, count in sorted(result.malformed_reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {count:4d}  {reason}")
    bad = [r for r in result.regions if not r.checksum_ok]
    if bad:
        lines.append(f"  !! {len(bad)} regions returned a bad checksum -- the path corrupted them")
    total = sum(r.size for r in result.regions)
    lines.append(f"  {total} bytes readable in total")
    return "\n".join(lines)


__all__ = [
    "Probe",
    "Region",
    "Stream",
    "SweepResult",
    "Sweeper",
    "Timing",
    "calibrate",
    "summarise",
    "asdict",
]
