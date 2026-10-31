# Zenodo release metadata

Zenodo archives GitHub releases using `CITATION.cff`. The release workflow then enriches the matching published Zenodo
record from `metadata.json`. It keeps the version DOI, archive, citation authors and unrelated metadata. The updater
adds affiliations only to creators matched by ORCID. The contact is taken from the citation author matching
`contact_orcid`. Additional contributors can be added to `contributors` with a Zenodo `person_or_org`, a personal name,
an optional ORCID and `role`. Joaquin Matres is listed as a supervisor.

The metadata includes the Aalto University ROR, documentation and PyPI links, English as the primary language, a Finnish
title and abstract, additional subjects and Python as a programming language. Review new translations, identifiers and
contributor roles before a release. `CITATION.cff` remains the source of citation authors, title and English abstract.

`metadata.schema.json` validates this updater's configuration and the InvenioRDM field shapes it uses. The upstream
record schema describes a complete Zenodo record, while this file is only an overlay. The pre-commit hook and updater
check the local schema; Zenodo still validates vocabulary identifiers and the complete record on publication.

Set the repository secret `ZENODO_ACCESS_TOKEN` to a Zenodo personal access token belonging to an account allowed to
edit the record. The documented deposit scopes are `deposit:write` and `deposit:actions`. A release is archived
asynchronously, so the updater waits up to 20 minutes for the exact tag, concept record and GitHub release URL. It edits
the published record's metadata, republishes it and verifies the result. Zenodo can briefly show the initial metadata
while this runs. If it fails, the release itself and its files remain published. Inspect the record and any existing
draft before retrying. Do not discard someone else's draft.

To retry an existing release, run the **Update Zenodo metadata** workflow manually and enter its tag. A read-only
preview needs no token:

```sh
uv run --script .github/zenodo/update_metadata.py v0.4.2 --dry-run --timeout 0
```

Sandbox has separate accounts, tokens and records. Copy `metadata.json` to a temporary file, change `concept_id`,
`anchor_record_id` and `repository` to match a sandbox GitHub archive, and use its tag and sandbox token:

```sh
ZENODO_ACCESS_TOKEN=... uv run --script .github/zenodo/update_metadata.py v1.2.3 --sandbox --config /path/to/sandbox-metadata.json
```

The API update has not been validated against a sandbox record yet. The token must never be printed or committed.
Inspect the published version and concept DOI metadata after testing a release.
