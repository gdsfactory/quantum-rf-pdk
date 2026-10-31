"""Tests for the release metadata updater."""

import copy
import importlib.util
import json
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
from jsonschema import exceptions as jsonschema_exceptions

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / ".github/zenodo/update_metadata.py"
SPEC = importlib.util.spec_from_file_location("zenodo_metadata", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
zenodo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(zenodo)
CONFIG = json.loads((SCRIPT.parent / "metadata.json").read_text())


def record() -> dict[str, Any]:
    return {
        "id": "123",
        "revision_id": 1,
        "parent": {"id": CONFIG["concept_id"]},
        "pids": {"doi": {"identifier": "10.5281/zenodo.123"}},
        "files": {"enabled": True},
        "metadata": {
            "version": "v1.2.3",
            "title": "Original title",
            "creators": [
                {
                    "person_or_org": {
                        "type": "personal",
                        "name": "Savola, Niko",
                        "identifiers": [
                            {"scheme": "orcid", "identifier": "0000-0002-8036-4984"}
                        ],
                    },
                    "affiliations": [{"name": "GDSFactory"}],
                }
            ],
            "related_identifiers": [
                {
                    "identifier": f"{CONFIG['repository']}/tree/v1.2.3",
                    "scheme": "url",
                    "relation_type": {"id": "issupplementto"},
                }
            ],
            "additional_descriptions": [
                {"description": "Existing note", "type": {"id": "notes"}}
            ],
            "subjects": [{"subject": "Existing subject"}],
        },
        "custom_fields": {
            "code:codeRepository": CONFIG["repository"],
            "other:field": "preserve",
        },
    }


def test_enrichment_preserves_record_and_is_idempotent() -> None:
    original = record()
    untouched = copy.deepcopy(original)
    enriched = zenodo.apply_metadata(original, CONFIG)
    assert original == untouched
    assert enriched["pids"] == original["pids"]
    assert enriched["files"] == original["files"]
    assert enriched["metadata"]["title"] == "Original title"
    assert enriched["custom_fields"]["other:field"] == "preserve"
    assert {
        a.get("name") for a in enriched["metadata"]["creators"][0]["affiliations"]
    } == {"GDSFactory", "Aalto University"}
    assert enriched["metadata"]["contributors"][0]["role"]["id"] == "contactperson"
    assert {
        x["type"]["id"] for x in enriched["metadata"]["additional_descriptions"]
    } == {"notes", "abstract"}
    assert zenodo.apply_metadata(enriched, CONFIG) == enriched


def test_matches_release_requires_concept_tag_and_github_reference() -> None:
    data = record()
    assert zenodo.matches_release(data, "v1.2.3", CONFIG)
    assert not zenodo.matches_release(data, "v1.2.4", CONFIG)
    data["parent"]["id"] = "different"
    assert not zenodo.matches_release(data, "v1.2.3", CONFIG)
    data["parent"]["id"] = CONFIG["concept_id"]
    data["metadata"]["related_identifiers"][0]["identifier"] = "https://example.com/"
    assert not zenodo.matches_release(data, "v1.2.3", CONFIG)


def test_existing_translation_is_not_overwritten() -> None:
    data = record()
    data["metadata"]["additional_descriptions"].append({
        "type": {"id": "abstract"},
        "lang": {"id": "fin"},
        "description": "Manually edited",
    })
    with pytest.raises(ValueError, match="different description"):
        zenodo.apply_metadata(data, CONFIG)


def test_updater_edits_existing_record_and_checks_public_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = record()
    calls = []
    state = {"record": original}

    def request(
        base: str,
        path: str,
        *,
        token: str = "",
        method: str = "GET",
        data: Any = None,
        if_match: str = "",
    ) -> dict[str, Any]:
        assert base == "https://zenodo.org"
        calls.append((method, path))
        if path.endswith("/draft") and method == "GET":
            raise urllib.error.HTTPError(path, 404, "not found", Message(), None)
        if method != "GET":
            assert token
        if method in {"PUT", "POST"} and if_match:
            assert if_match == "1"
        if method == "PUT":
            state["record"] = data
        return state["record"]

    monkeypatch.setattr(zenodo, "request_json", request)
    zenodo.update("https://zenodo.org", original, "test-token", CONFIG, dry_run=False)
    assert calls == [
        ("GET", "/api/records/123/draft"),
        ("POST", "/api/records/123/draft"),
        ("PUT", "/api/records/123/draft"),
        ("POST", "/api/records/123/draft/actions/publish"),
        ("GET", "/api/records/123"),
    ]
    assert state["record"]["pids"] == original["pids"]
    calls.clear()
    zenodo.update(
        "https://zenodo.org", state["record"], "test-token", CONFIG, dry_run=False
    )
    assert calls == []


def test_unexpected_draft_is_not_published(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def request(
        base: str,
        path: str,
        *,
        token: str = "",
        method: str = "GET",
        data: Any = None,
        if_match: str = "",
    ) -> dict[str, Any]:
        assert (base, path, token, data) == (
            "https://zenodo.org",
            "/api/records/123/draft",
            "test-token",
            None,
        )
        calls.append(method)
        assert not if_match
        if method == "GET":
            raise urllib.error.HTTPError(path, 404, "not found", Message(), None)
        return {"id": "999", "parent": {"id": CONFIG["concept_id"]}}

    monkeypatch.setattr(zenodo, "request_json", request)
    with pytest.raises(ValueError, match="Unexpected draft identity"):
        zenodo.update(
            "https://zenodo.org", record(), "test-token", CONFIG, dry_run=False
        )
    assert calls == ["GET", "POST"]


def test_existing_draft_is_not_modified(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def request(
        base: str,
        path: str,
        *,
        token: str = "",
        method: str = "GET",
        data: Any = None,
        if_match: str = "",
    ) -> dict[str, Any]:
        assert (base, path, token, method, data, if_match) == (
            "https://zenodo.org",
            "/api/records/123/draft",
            "test-token",
            "GET",
            None,
            "",
        )
        calls.append(method)
        return record()

    monkeypatch.setattr(zenodo, "request_json", request)
    with pytest.raises(RuntimeError, match="existing draft"):
        zenodo.update(
            "https://zenodo.org", record(), "test-token", CONFIG, dry_run=False
        )
    assert calls == ["GET"]


def test_search_checks_all_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    matching = record()
    other = copy.deepcopy(matching)
    other["metadata"]["version"] = "v0.0.1"
    requests = []

    def request(base: str, path: str) -> dict[str, Any]:
        assert base == "https://zenodo.org"
        requests.append(path)
        hits = [other] * 25 if path.endswith("page=1") else [matching]
        return {"hits": {"hits": hits}}

    monkeypatch.setattr(zenodo, "request_json", request)
    assert zenodo.find_record("https://zenodo.org", "v1.2.3", CONFIG) == matching
    assert len(requests) == 2


def test_dry_run_does_not_write(monkeypatch: pytest.MonkeyPatch) -> None:
    def request(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("dry run must not write")

    monkeypatch.setattr(zenodo, "request_json", request)
    zenodo.update("https://zenodo.org", record(), "", CONFIG, dry_run=True)


def test_supervisor_is_a_contributor_and_is_not_duplicated() -> None:
    enriched = zenodo.apply_metadata(record(), CONFIG)
    supervisor = CONFIG["contributors"][0]
    assert supervisor["person_or_org"]["given_name"] == "Joaquin"
    assert supervisor["person_or_org"]["family_name"] == "Matres"
    assert zenodo.orcid_of(supervisor) == "0009-0001-1601-6109"
    assert supervisor["role"]["id"] == "supervisor"
    assert supervisor in enriched["metadata"]["contributors"]
    assert CONFIG["contributors"][1] in enriched["metadata"]["contributors"]
    assert len(enriched["metadata"]["creators"]) == 1
    assert zenodo.apply_metadata(enriched, CONFIG) == enriched


def test_metadata_schema_accepts_config_and_rejects_invalid_contributors() -> None:
    schema = json.loads(zenodo.SCHEMA.read_text())
    validator = zenodo.Draft202012Validator(schema)
    validator.check_schema(schema)
    validator.validate(CONFIG)
    invalid = copy.deepcopy(CONFIG)
    del invalid["contributors"][0]["role"]
    with pytest.raises(jsonschema_exceptions.ValidationError):
        validator.validate(invalid)


def test_supervisor_orcid_updates_existing_contributor() -> None:
    enriched = zenodo.apply_metadata(record(), CONFIG)
    contributor = enriched["metadata"]["contributors"][1]
    contributor["person_or_org"]["identifiers"] = [
        {"scheme": "other", "identifier": "existing"}
    ]
    updated = zenodo.apply_metadata(enriched, CONFIG)
    assert len(updated["metadata"]["contributors"]) == 3
    assert updated["metadata"]["contributors"][1]["person_or_org"]["identifiers"] == [
        {"scheme": "other", "identifier": "existing"},
        {"scheme": "orcid", "identifier": "0009-0001-1601-6109"},
    ]
    assert zenodo.apply_metadata(updated, CONFIG) == updated


def test_translated_title_language_and_pypi_link() -> None:
    data = record()
    data["metadata"]["additional_titles"] = [
        {
            "title": "An existing subtitle",
            "type": {"id": "subtitle"},
            "lang": {"id": "eng"},
        }
    ]
    enriched = zenodo.apply_metadata(data, CONFIG)
    metadata = enriched["metadata"]
    assert {item["id"] for item in metadata["languages"]} == {"eng"}
    assert metadata["additional_titles"] == [
        data["metadata"]["additional_titles"][0],
        {
            "title": "QPDK: Suprajohtavien kvanttipiirien prosessisuunnittelupaketti",
            "type": {"id": "translated-title"},
            "lang": {"id": "fin"},
        },
    ]
    assert {
        "identifier": "https://pypi.org/project/qpdk/",
        "scheme": "url",
        "relation_type": {"id": "issupplementedby"},
    } in metadata["related_identifiers"]
    assert zenodo.apply_metadata(enriched, CONFIG) == enriched


def test_existing_finnish_title_is_not_overwritten() -> None:
    data = record()
    data["metadata"]["additional_titles"] = [
        {
            "title": "Manually edited",
            "type": {"id": "translated-title"},
            "lang": {"id": "fin"},
        }
    ]
    with pytest.raises(ValueError, match="different title"):
        zenodo.apply_metadata(data, CONFIG)
