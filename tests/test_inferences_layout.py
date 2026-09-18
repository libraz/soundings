"""The fence between the archive and what was read out of it, gated.

Everything here is about keeping two kinds of claim apart and keeping a
retraction cheap. The prose in `inferences/LICENSE` states the arrangement; these
are the parts of it that a file can quietly break.

The one that matters most is the first. References run one way: an inference
cites a record and no record cites an inference. That is what lets a consumer
harvest `data/` and get measurements only, without reading a word of this
project's reasoning -- and it is the only protection that works on a consumer who
never reads any of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soundings import inferences

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "inferences"
SCHEMA = json.loads((HERE / "schema" / "inference.json").read_text())
CLAIMS = inferences.every(ROOT)


def test_there_is_something_to_check():
    """A suite that passes on an empty directory says nothing about a full one."""
    assert CLAIMS, "no inferences to check; every gate below would pass vacuously"


def test_no_record_cites_an_inference():
    """The fence. One way only, and it is checked by content and not by intent."""
    assert inferences.citing_an_inference(ROOT) == []


def test_the_dedication_is_a_separate_file():
    """Same dedication, separate file, because the reason is not a legal one."""
    assert (HERE / "LICENSE").is_file()
    assert (ROOT / "data" / "LICENSE").is_file()


@pytest.mark.parametrize("path", CLAIMS, ids=lambda p: p.name)
def test_a_claim_carries_its_envelope(path: Path):
    claim = inferences.load(path)
    for key in SCHEMA["required"]:
        assert key in claim, f"{path.name} has no {key}"
    head = claim["inference"]
    for key in SCHEMA["properties"]["inference"]["required"]:
        assert key in head, f"{path.name} envelope has no {key}"
    assert head["schema_version"] == inferences.SCHEMA_VERSION
    assert head["state"] in set(SCHEMA["properties"]["inference"]["properties"]["state"]["enum"])
    # One level further in than this used to reach. `about` is how a reader or a
    # query asks whether a claim covers the type in front of them, and a claim short
    # of one of its keys answers that question with nothing -- which is not what the
    # claim says, and which took `soundings inferences closed` down over every other
    # claim rather than being caught here, where a defect in a claim belongs.
    about = head["about"]
    for key in SCHEMA["properties"]["inference"]["properties"]["about"]["required"]:
        assert key in about, (
            f"{path.name} has an `about` block with no {key!r}. A claim that cannot say "
            "which types and which addresses it is about cannot be checked against a "
            "unit by anything except a reader."
        )


@pytest.mark.parametrize("path", CLAIMS, ids=lambda p: p.name)
def test_a_claim_says_what_it_adds(path: Path):
    """Required and never empty.

    A claim that adds nothing to the records it cites is a restatement of a
    measurement and belongs where the measurement is. This field is the whole of
    what makes a file here an inference rather than a copy.
    """
    claim = inferences.load(path)
    assert claim["adds"].strip(), f"{path.name} does not say what it adds to its evidence"


@pytest.mark.parametrize("path", CLAIMS, ids=lambda p: p.name)
def test_a_claim_says_what_would_refute_it(path: Path):
    claim = inferences.load(path)
    assert claim["refuted_by"].strip(), f"{path.name} cannot say what would show it wrong"


@pytest.mark.parametrize("path", CLAIMS, ids=lambda p: p.name)
def test_an_untested_claim_says_what_could_have_contradicted_it(path: Path):
    """The rule that keeps the era and the forum from closing anything.

    A claim resting on what was buildable at the time, or on somebody's report, is
    worth reading only if it can name the measurement the unit could have answered
    it with. Without that it says nothing about this unit at all, however
    plausible it is.
    """
    claim = inferences.load(path)
    if claim["inference"]["state"] != "standing_untested":
        return
    assert claim.get("could_have_been_refuted_by"), (
        f"{path.name} stands on priors and cannot name a measurement that could have "
        "contradicted it"
    )


@pytest.mark.parametrize("path", CLAIMS, ids=lambda p: p.name)
def test_a_standing_alternative_can_be_settled_or_says_why_not(path: Path):
    """An alternative left standing owes either a way to separate it or a reason.

    A margin below one means the measurement could not have told the two apart,
    and queueing it would put a run on the list that cannot answer -- the negative
    with no stated sensitivity that this project refuses to publish, wearing the
    clothes of a plan.
    """
    claim = inferences.load(path)
    for alternative in claim["alternatives"]:
        if not alternative.get("standing") or alternative.get("equivalent_under_this_test"):
            continue
        separated = alternative.get("separated_by")
        assert separated is not None, (
            f"{path.name}: {alternative['reading'][:50]!r} stands with no way to settle it"
        )
        if separated.get("why_not_separable"):
            continue
        assert separated.get("margin", 0) >= 1, (
            f"{path.name}: {alternative['reading'][:50]!r} is queued behind a measurement "
            "that cannot separate it"
        )


@pytest.mark.parametrize("path", CLAIMS, ids=lambda p: p.name)
def test_an_alternative_that_stopped_standing_says_which_way_it_went(path: Path):
    """An alternative ends three ways and the file has to say which.

    It is refuted, it is still standing, or **the measurement found for it and the
    claim took it up**. The third is not the first: a key named `ruled_out_by`
    carrying a sentence about a reading that turned out to be right is the kind of
    thing this archive exists to refuse, and a reader who only skimmed the key names
    would come away with the opposite of what was measured.
    """
    claim = inferences.load(path)
    for alternative in claim["alternatives"]:
        if alternative.get("standing"):
            continue
        fell = alternative.get("ruled_out_by")
        taken = alternative.get("taken_up_by")
        assert fell or taken, (
            f"{path.name}: {alternative['reading'][:50]!r} is not standing and does not say why"
        )
        assert not (fell and taken), (
            f"{path.name}: {alternative['reading'][:50]!r} says it was both refuted and taken up"
        )


@pytest.mark.parametrize("path", CLAIMS, ids=lambda p: p.name)
def test_community_evidence_says_how_far_it_can_be_traced(path: Path):
    """`hearsay` is a usable value and a missing one is not.

    A report whose grounding is unknown recorded as unknown is a weak claim a
    reader can weigh. The same report with the field left off reads as better than
    it is, and there is nothing in the file to say otherwise.
    """
    claim = inferences.load(path)
    for reported in claim["rests_on"]["community"]:
        assert reported.get("source"), f"{path.name}: a community claim with no source"
        assert reported.get("traceable_to") in {
            "measurement", "rom", "documentation", "hearsay", "unknown"
        }
        assert reported.get("we_have_not_verified") is True


@pytest.mark.parametrize("path", CLAIMS, ids=lambda p: p.name)
def test_an_era_prior_says_what_would_be_wrong(path: Path):
    claim = inferences.load(path)
    for prior in claim["rests_on"]["era_priors"]:
        assert prior.get("would_be_wrong_if", "").strip(), (
            f"{path.name}: a prior with no refutation is a habit of thought"
        )


@pytest.mark.parametrize("path", CLAIMS, ids=lambda p: p.name)
def test_every_cited_record_is_in_the_archive(path: Path):
    claim = inferences.load(path)
    for citation in claim["rests_on"]["measurements"]:
        assert (ROOT / citation["file"]).is_file(), (
            f"{path.name} cites {citation['file']}, which is not here"
        )
        assert len(citation["keys"]) == len(citation["values"]), (
            f"{path.name} cites {citation['file']} with keys and values that do not pair"
        )


def test_no_claim_rests_on_a_figure_that_has_moved():
    """The quiet failure this whole arrangement exists to catch.

    A record is re-published, a number changes, and the conclusion drawn from the
    old one goes on standing. It has happened twice in this archive already, once
    for thirty-nine decibels of floor and once for a decay time that became a
    null, and on both occasions nothing failed.
    """
    moved = inferences.stale(ROOT)
    assert moved == [], "\n".join(
        f"{m['inference']}: {m['file']} {m.get('key', '')} was {m.get('was')} now {m.get('now')}"
        for m in moved
    )


def test_a_supersession_points_somewhere():
    for path in CLAIMS:
        claim = inferences.load(path)
        head = claim["inference"]
        if head["state"] != "superseded":
            continue
        target = head.get("superseded_by")
        assert target and (HERE / head["unit_id"] / target).is_file(), (
            f"{path.name} is superseded by a file that is not here"
        )


def test_no_claim_has_run_past_its_rounds():
    """Three revisions and then it is parked, keeping what it found.

    The ceiling is not about giving up. It is about one type not absorbing the
    project, which is how a session ends with a pile of curiosities and no stage
    finished.
    """
    for path in CLAIMS:
        head = inferences.load(path)["inference"]
        if head.get("rounds", 0) <= inferences.ROUND_CEILING:
            continue
        assert head["state"] == "parked", (
            f"{path.name} has had {head['rounds']} rounds and is not parked"
        )


@pytest.mark.parametrize("path", CLAIMS, ids=lambda p: p.name)
def test_a_verdict_is_one_of_the_words_the_schema_names(path: Path):
    """The field two queries print, held to the six values it is allowed.

    A verdict is not prose. `inferences.closed` puts it in a listing and the
    readings query prints it, so a claim answering with anything else prints
    something else where every other claim prints a word -- and a reader comparing
    two claims is comparing a word against an object.

    The temptation is real and one claim took it: a reading that holds over part of
    a byte's range wants to say so, and keying the verdict by the range it covers
    reads better than a word does. Where it holds goes beside the verdict, not
    inside it. Nothing failed for the day that claim sat there, because the schema
    states the enum and nothing was checking claims against it.
    """
    verdict = (inferences.load(path).get("reproduces") or {}).get("verdict")
    if verdict is None:
        return
    allowed = SCHEMA["$defs"]["reproduces"]["properties"]["verdict"]["enum"]
    assert verdict in allowed, (
        f"{path.name} has a verdict of {verdict!r}, which is not one of {allowed}. "
        "Where a verdict holds is said beside it and not inside it."
    )


def _prose_lists(node, path: str):
    """Every list in a claim that holds prose, with where in the claim it sits."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _prose_lists(value, f"{path}.{key}")
    elif isinstance(node, list):
        # Below about thirty characters a repeated string is a value -- an address,
        # a verdict, a byte -- and a list repeating one of those is ordinary.
        if any(isinstance(x, str) and len(x) > 30 for x in node):
            yield path, node
        for index, value in enumerate(node):
            yield from _prose_lists(value, f"{path}[{index}]")


@pytest.mark.parametrize("path", CLAIMS, ids=lambda p: p.name)
def test_a_claim_does_not_say_the_same_thing_twice(path: Path):
    """What a round script that ran twice leaves behind, and it leaves it silently.

    A round edits a published claim in place: it appends its name to `made_by`,
    appends what it raised to `what_this_does_not_settle`, and rewrites the
    figures it moved. Rewriting twice lands on the same value and nothing shows.
    Appending twice does not, and it had happened four times here before anything
    looked -- one claim carrying the same open item three times, and one carrying
    a round's three scripts five times each.

    The cost is not tidiness. An open item printed three times is a reader being
    told three times that the same thing is unsettled, and a count of what is open
    that is wrong by two; a name twice in the provenance says a script ran twice
    and not what it did.
    """
    claim = inferences.load(path)

    names = [
        name.strip()
        for name in (claim["inference"].get("made_by") or "").split(",")
        if name.strip()
    ]
    assert len(set(names)) == len(names), (
        f"{path.name} names a script more than once in `made_by`. A round script is "
        "not idempotent unless it drops its own entries before writing them again."
    )

    for where, items in _prose_lists(claim, ""):
        strings = [x for x in items if isinstance(x, str)]
        if len(strings) != len(items):
            continue
        assert len(set(strings)) == len(strings), (
            f"{path.name} holds the same entry more than once in {where or 'the claim'}"
        )


def test_the_listing_is_the_files():
    for unit in sorted(p.name for p in HERE.glob("*/") if (p / "index.json").is_file()):
        listing = json.loads((HERE / unit / "index.json").read_text())
        assert listing == inferences.index(ROOT, unit), (
            f"inferences/{unit}/index.json disagrees with the files it is generated from"
        )


def test_a_model_names_the_class_it_is_a_candidate_in():
    """A candidate with no catalogue cannot have a decoy, and a decoy is a gate."""
    classes = {
        json.loads(p.read_text())["class"]
        for p in (HERE / "candidates").glob("*.json")
    }
    for path in sorted((HERE / "models").glob("*.json")):
        model = json.loads(path.read_text())["model"]
        assert model["class"] in classes, f"{path.name} is in no catalogue"
        catalogue = next(
            json.loads(p.read_text())
            for p in (HERE / "candidates").glob("*.json")
            if json.loads(p.read_text())["class"] == model["class"]
        )
        assert len(catalogue["candidates"]) >= 2, (
            f"{model['class']} holds one candidate, so nothing can be its decoy"
        )


def test_a_generated_comparison_set_is_still_what_the_index_yields():
    """The quiet failure the staleness query cannot see.

    A claim's comparison set is generated so that nobody chooses which readings a
    model is held against. Once generated it is committed, and records go on being
    published under the same type -- so the set the claim names drifts away from
    the set its own rule yields, and nothing fails. Three frequency records did
    exactly that: they turned the equaliser's breakdown gate over, and the claim
    carrying the old verdict passed every check in this file, because they were
    records it did not cite.

    So the rule is written down as a rule and re-applied here. A claim whose set
    cannot be expressed as one says why, which is a thing to read rather than a
    thing to pass.
    """
    for path in sorted(HERE.glob("*/*.json")):
        if path.name == "index.json":
            continue
        claim = json.loads(path.read_text())
        against = (claim.get("reproduces") or {}).get("compared_against")
        if not against or not against.get("records"):
            continue
        where = f"{path.parent.name}/{path.name}"
        spec = against.get("filter")
        if spec is None:
            assert (against.get("why_not_a_filter") or "").strip(), (
                f"{where} names a comparison set and neither generates it from a "
                "filter nor says why it cannot"
            )
            continue
        # What the filter yields and the scoring could not read are two different
        # lists, and both are the claim's to carry. A record the run dropped is
        # named with the reason rather than left out of the total, so that a set
        # that shrank and a set that was never that big do not read the same.
        dropped = against.get("then_dropped") or []
        for item in dropped:
            assert item.get("file") and (item.get("why") or "").strip(), (
                f"{where} drops a record from its comparison set without saying which or why"
            )
        named = sorted(against["records"] + [item["file"] for item in dropped])
        assert inferences.comparison_set(ROOT, spec) == named, (
            f"{where} was scored against a set the index no longer yields -- "
            "re-run the scoring and republish the claim"
        )


def test_what_is_not_chased_says_how_each_was_bounded():
    """The list that keeps `not explained` and `not required` apart.

    Without it the two read the same, and the second gets worked on as though it
    were the first -- which is the endless part of this.
    """
    listed = json.loads((HERE / "not-chased.json").read_text())
    assert listed["not_chased"], "nothing is listed as out of the model's reach"
    for item in listed["not_chased"]:
        assert item["what"].strip()
        assert item["how_it_was_bounded"].strip(), (
            f"{item['what']!r} is not chased and does not say what bounds it"
        )


# ---- a citation names a row by one of the row's own fields


def test_a_row_named_by_a_take_resolves_through_the_dot_in_the_file_name():
    """Otherwise a claim resting on a figure that is really there reads as stale.

    A record names some of its rows by the take they came from, and a take is a
    file name with a dot in it. Cutting a dotted key on every dot puts half that
    name in one step and half in the next, so the citation comes back
    unresolvable -- which the stale query reports as a figure that has moved.
    """
    record = {
        "control": {
            "readings": [
                {"take": "held-126-bypassed-00-00.wav", "heard_db": -53.02},
                {"take": "held-126-bypassed-01-00.wav", "heard_db": -53.01},
            ]
        }
    }
    key = "control.readings[take=held-126-bypassed-01-00.wav].heard_db"
    assert inferences.resolve(record, key) == -53.01


def test_a_row_named_by_a_number_with_a_decimal_point_resolves_too():
    """The same cut, and the one a sweep's own figures run into first."""
    record = {"readings": [{"largest_db": 0.37, "at": "a"}, {"largest_db": -1.5, "at": "b"}]}
    assert inferences.resolve(record, "readings[largest_db=-1.5].at") == "b"


def test_an_ordinary_dotted_key_is_still_cut_on_its_dots():
    record = {"reference": {"heard_floor_db": 0.01}}
    assert inferences.resolve(record, "reference.heard_floor_db") == 0.01


def test_a_row_can_be_named_by_more_than_one_of_its_fields():
    """A record whose rows are one byte written from two places has two rows saying
    `value=32`, and they answer differently -- which is the whole reason it exists.
    A citation that named a row by the byte alone would point at whichever came
    first, and that is the silent repointing this naming is here to prevent."""
    record = {
        "readings": [
            {"value": 32, "came_from": "rest", "rate_hz": 1.4071},
            {"value": 32, "came_from": "127", "rate_hz": 1.6520},
        ]
    }
    assert inferences.resolve(
        record, "readings[value=32,came_from=rest].rate_hz"
    ) == 1.4071
    assert inferences.resolve(
        record, "readings[value=32,came_from=127].rate_hz"
    ) == 1.6520


def test_one_field_still_names_a_row():
    record = {"readings": [{"value": 52, "largest_db": -3.0}]}
    assert inferences.resolve(record, "readings[value=52].largest_db") == -3.0


def _listing(tmp_path: Path, unit: str, stage: str, entries: list[dict]) -> Path:
    where = tmp_path / "data" / "units" / unit
    where.mkdir(parents=True)
    (where / "index.json").write_text(
        json.dumps({"unit_id": unit, "records": len(entries), "stages": {stage: entries}})
    )
    return tmp_path


RUN = {"stage": "efx-rate", "types": ["11 07", "01 22"],
       "addresses": ["40 03 08", "40 03 03"]}

BOTH_SIDES = [
    {"file": "efx-rate/on-the-multi.json", "about": {"type": "11 07", "address": "40 03 08"}},
    {"file": "efx-rate/on-the-single.json", "about": {"type": "01 22", "address": "40 03 03"}},
]


def test_a_queued_run_whose_records_exist_and_are_uncited_is_flagged(tmp_path: Path):
    """The queue is derived from the claims, so a made run stays on it.

    Between the takes landing and the fold-in being written the queue offers a run
    that exists, and that gap has been long enough to book the unit against twice.
    Nothing here reads a key saying the run was made: that key is the thing that
    goes stale.
    """
    root = _listing(tmp_path, "a-unit", "efx-rate", BOTH_SIDES)
    found = inferences.already_recorded(root, "a-unit", RUN, {"cites": []})
    assert found is not None
    assert [side["side"] for side in found["sides"]] == [
        "11 07 40 03 08", "01 22 40 03 03",
    ]


def test_one_side_already_recorded_is_not_enough(tmp_path: Path):
    """One side of a pair usually exists before the run, which is often why the
    pair was chosen at all. Flagging on that would empty the queue of exactly the
    entries that were best thought through."""
    root = _listing(tmp_path, "a-unit", "efx-rate", BOTH_SIDES[1:])
    assert inferences.already_recorded(root, "a-unit", RUN, {"cites": []}) is None


def test_a_run_whose_records_the_claim_already_cites_is_not_flagged(tmp_path: Path):
    """Folded in. The entry is then a queue item nobody cleared, which is a
    different defect and not this one's to report."""
    root = _listing(tmp_path, "a-unit", "efx-rate", BOTH_SIDES)
    claim = {"cites": [entry["file"] for entry in BOTH_SIDES]}
    assert inferences.already_recorded(root, "a-unit", RUN, claim) is None


def test_a_run_that_does_not_say_what_it_is_about_cannot_be_looked_up(tmp_path: Path):
    """And says so rather than guessing. A survey stage whose subject is a list
    is the case this protects: it has no one type and no one address, and a
    lookup that fell back to the stage alone would match every record it wrote."""
    root = _listing(tmp_path, "a-unit", "efx-rate", BOTH_SIDES)
    vague = {"stage": "efx-rate", "minutes": 20}
    assert inferences.already_recorded(root, "a-unit", vague, {"cites": []}) is None
