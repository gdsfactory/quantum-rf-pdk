#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["jsonschema==4.26.0"]
# ///
"""Enrich an existing Zenodo GitHub release without changing its files or DOI."""

import argparse
import copy
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

LOG = logging.getLogger(__name__)
ACCEPT = "application/vnd.inveniordm.v1+json"
CONFIG = Path(__file__).with_name("metadata.json")
SCHEMA = Path(__file__).with_name("metadata.schema.json")


def request_json(
    base: str,
    path: str,
    *,
    token: str = "",
    method: str = "GET",
    data: Any = None,
    if_match: str = "",
) -> Any:
    """Send a request to the selected Zenodo instance."""
    if base not in {
        "https://zenodo.org",
        "https://sandbox.zenodo.org",
    } or not path.startswith("/api/"):
        raise ValueError("Invalid Zenodo API URL")
    headers = {"Accept": ACCEPT}
    if if_match:
        headers["If-Match"] = if_match
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(data).encode()
    request = urllib.request.Request(  # ruff: ignore[suspicious-url-open-usage]
        base + path, data=payload, method=method, headers=headers
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # ruff: ignore[suspicious-url-open-usage]
        return json.load(response)


def orcid_of(person: dict[str, Any]) -> str:
    """Read the ORCID from a Zenodo creator or contributor."""
    for identifier in person.get("person_or_org", {}).get("identifiers", []):
        if identifier.get("scheme") == "orcid":
            return identifier["identifier"].removeprefix("https://orcid.org/")
    return ""


def same_contributor(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Match contributors by ORCID when available, otherwise by name and role."""
    if not left.get("role", {}).get("id") or left.get("role", {}).get(
        "id"
    ) != right.get("role", {}).get("id"):
        return False
    left_orcid, right_orcid = orcid_of(left), orcid_of(right)
    if left_orcid and right_orcid:
        return left_orcid == right_orcid

    def name(item: dict[str, Any]) -> str:
        person = item.get("person_or_org", {})
        if person.get("type") != "personal":
            return ""
        given = person.get("given_name", "").strip()
        family = person.get("family_name", "").strip()
        return f"{family}, {given}".casefold() if given and family else ""

    return bool(name(left)) and name(left) == name(right)


def matches_release(record: dict[str, Any], tag: str, config: dict[str, Any]) -> bool:
    """Require the same concept, version and GitHub release URL."""
    if record.get("parent", {}).get("id") != config["concept_id"]:
        return False
    metadata = record.get("metadata", {})
    expected = f"{config['repository']}/tree/{tag}"
    return metadata.get("version") == tag and any(
        item.get("identifier") == expected
        and item.get("relation_type", {}).get("id") == "issupplementto"
        for item in metadata.get("related_identifiers", [])
    )


def find_record(base: str, tag: str, config: dict[str, Any]) -> dict[str, Any] | None:
    """Search all versions of the known record, not the search index's latest result."""
    page = 1
    while True:
        result = request_json(
            base,
            f"/api/records/{config['anchor_record_id']}/versions?size=25&page={page}",
        )
        hits = result["hits"]["hits"]
        for record in hits:
            if matches_release(record, tag, config):
                return record
        if len(hits) < 25:
            return None
        page += 1


def apply_metadata(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Add curated fields while keeping the existing citation and unrelated metadata."""
    result = copy.deepcopy(record)
    metadata = result["metadata"]
    for creator in metadata["creators"]:
        orcid = orcid_of(creator)
        if orcid not in config["affiliations"]:
            continue
        affiliations = creator.setdefault("affiliations", [])
        for entry in config["affiliations"][orcid]:
            if not any(
                (entry.get("id") and item.get("id") == entry["id"])
                or item.get("name", "").casefold() == entry.get("name", "").casefold()
                for item in affiliations
            ):
                affiliations.append(entry)

    contact = next(
        (c for c in metadata["creators"] if orcid_of(c) == config["contact_orcid"]),
        None,
    )
    if contact is None:
        raise ValueError("The contact ORCID is not a creator in this release")
    contributors = metadata.setdefault("contributors", [])
    if not any(
        orcid_of(c) == config["contact_orcid"]
        and c.get("role", {}).get("id") == "contactperson"
        for c in contributors
    ):
        contributors.append({
            "person_or_org": copy.deepcopy(contact["person_or_org"]),
            "role": {"id": "contactperson"},
            "affiliations": copy.deepcopy(contact.get("affiliations", [])),
        })

    for entry in config.get("contributors", []):
        if not same_contributor(entry, entry):
            raise ValueError("Configured contributors need a name or ORCID and role")
        existing = next(
            (item for item in contributors if same_contributor(item, entry)), None
        )
        if existing is None:
            contributors.append(copy.deepcopy(entry))
        elif not orcid_of(existing) and orcid_of(entry):
            identifiers = existing["person_or_org"].setdefault("identifiers", [])
            identifiers.extend(copy.deepcopy(entry["person_or_org"]["identifiers"]))

    languages = metadata.setdefault("languages", [])
    for entry in config["languages"]:
        if not any(item.get("id") == entry["id"] for item in languages):
            languages.append(entry)
    titles = metadata.setdefault("additional_titles", [])
    for entry in config["additional_titles"]:
        matches = [
            item
            for item in titles
            if item.get("type", {}).get("id") == entry["type"]["id"]
            and item.get("lang", {}).get("id") == entry["lang"]["id"]
        ]
        if not matches:
            titles.append(entry)
        elif not any(item.get("title") == entry["title"] for item in matches):
            raise ValueError(
                "A different title already exists for the same type and language"
            )
    subjects = metadata.setdefault("subjects", [])
    for name in config["subjects"]:
        if not any(
            item.get("subject", "").casefold() == name.casefold() for item in subjects
        ):
            subjects.append({"subject": name})
    descriptions = metadata.setdefault("additional_descriptions", [])
    for entry in config["additional_descriptions"]:
        matches = [
            item
            for item in descriptions
            if item.get("type", {}).get("id") == entry["type"]["id"]
            and item.get("lang", {}).get("id") == entry["lang"]["id"]
        ]
        if not matches:
            descriptions.append(entry)
        elif not any(
            item.get("description") == entry["description"] for item in matches
        ):
            raise ValueError(
                "A different description already exists for the same type and language"
            )
    relations = metadata.setdefault("related_identifiers", [])
    for entry in config["related_identifiers"]:
        if not any(
            item.get("identifier") == entry["identifier"]
            and item.get("relation_type", {}).get("id") == entry["relation_type"]["id"]
            for item in relations
        ):
            relations.append(entry)
    custom_fields = result.setdefault("custom_fields", {})
    for key, value in config["custom_fields"].items():
        if key == "code:programmingLanguage":
            existing = custom_fields.setdefault(key, [])
            for entry in value:
                if not any(item.get("id") == entry["id"] for item in existing):
                    existing.append(entry)
        elif key not in custom_fields:
            custom_fields[key] = value
        elif custom_fields[key] != value:
            raise ValueError(f"Existing {key} differs from the configured value")
    return result


def wait_for_record(
    base: str, tag: str, config: dict[str, Any], timeout: int
) -> dict[str, Any]:
    """Wait for the exact release to be indexed by Zenodo."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            record = find_record(base, tag, config)
            if record is not None:
                return record
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            LOG.warning("Zenodo lookup failed: HTTP %s", exc.code)
        except (urllib.error.URLError, TimeoutError) as exc:
            LOG.warning("Zenodo lookup failed: %s", exc)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Zenodo did not publish a matching record for {tag}")
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def update(
    base: str,
    record: dict[str, Any],
    token: str,
    config: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    """Edit and republish metadata, then check the published record."""
    tag = record.get("metadata", {}).get("version", "")
    if not matches_release(record, tag, config):
        raise ValueError("Record does not match the configured release")
    updated = apply_metadata(record, config)
    if updated == record:
        LOG.info("Record %s already contains the configured metadata", record["id"])
        return
    if dry_run:
        LOG.info(
            "Proposed metadata for record %s:\n%s",
            record["id"],
            json.dumps(
                {
                    "metadata": updated["metadata"],
                    "custom_fields": updated.get("custom_fields", {}),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return
    if not token:
        raise ValueError("ZENODO_ACCESS_TOKEN is required")
    record_path = f"/api/records/{record['id']}"
    try:
        request_json(base, record_path + "/draft", token=token)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    else:
        raise RuntimeError("An existing draft needs manual review before updating")
    draft = request_json(base, record_path + "/draft", token=token, method="POST")
    if draft.get("id") != record["id"] or not matches_release(draft, tag, config):
        raise ValueError("Unexpected draft identity")
    updated_draft = apply_metadata(draft, config)
    saved = request_json(
        base,
        record_path + "/draft",
        token=token,
        method="PUT",
        data=updated_draft,
        if_match=str(draft["revision_id"]),
    )
    request_json(
        base,
        record_path + "/draft/actions/publish",
        token=token,
        method="POST",
        if_match=str(saved["revision_id"]),
    )
    for attempt in range(12):
        current = request_json(base, record_path)
        if (
            current.get("parent", {}).get("id") == config["concept_id"]
            and apply_metadata(current, config) == current
        ):
            LOG.info("Verified metadata for record %s", record["id"])
            return
        if attempt < 11:
            time.sleep(5)
    raise RuntimeError(f"Could not verify published metadata for record {record['id']}")


def main() -> None:
    """Run the updater from a release workflow or manually."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="GitHub release tag, for example v0.4.2")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"v\d+\.\d+\.\d+", args.tag):
        parser.error("tag must be vMAJOR.MINOR.PATCH")
    if args.timeout < 0:
        parser.error("timeout must be non-negative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = json.loads(args.config.read_text())
    Draft202012Validator(json.loads(SCHEMA.read_text())).validate(config)
    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
    record = wait_for_record(base, args.tag, config, args.timeout)
    update(
        base,
        record,
        os.environ.get("ZENODO_ACCESS_TOKEN", ""),
        config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
