"""Offline standard-library regressions for contributor navigation checks."""

from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path

API = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_research_docs.py"))
check_documents = API["check_documents"]
load_inventory = API["load_inventory"]
issue_fields = API["issue_fields"]


class DocumentationChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name: str, text: str = "# Example\n") -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def check(self, text: str, known: set[str] | None = None):
        self.write("README.md", text)
        return check_documents(self.root, ("README.md",), known)

    def test_existing_target(self) -> None:
        self.write("docs/start.md")
        result = self.check("[Start](docs/start.md)")
        self.assertEqual(result.errors, [])
        self.assertEqual(result.checked_links, 1)

    def test_missing_target(self) -> None:
        self.assertIn("Missing local target", self.check("[Missing](absent.md)").errors[0])

    def test_missing_managed_document(self) -> None:
        result = check_documents(self.root, ("absent.md",))
        self.assertEqual(len(result.errors), 1)

    def test_fragment_is_not_claimed_validated(self) -> None:
        result = self.check("[Heading](#not-validated)")
        self.assertEqual(result.errors, [])
        self.assertEqual(result.checked_links, 1)

    def test_external_links_not_requested(self) -> None:
        result = self.check("[Site](https://example.invalid) [Mail](mailto:test@example.invalid)")
        self.assertEqual(result.errors, [])
        self.assertEqual(result.checked_links, 0)

    def test_fenced_and_inline_code_ignored(self) -> None:
        result = self.check("```md\n[x](absent.md)\n```\n`[x](absent.md)`\n~~~\n[y](other.md)\n~~~")
        self.assertEqual(result.errors, [])
        self.assertEqual(result.checked_links, 0)

    def test_image_target_checked(self) -> None:
        self.assertEqual(len(self.check("![chart](missing.svg)").errors), 1)

    def test_encoded_space(self) -> None:
        self.write("a file.md")
        self.assertEqual(self.check("[Target](a%20file.md)").errors, [])

    def test_escape_rejected(self) -> None:
        self.assertIn("escapes repository", self.check("[Outside](../outside.md)").errors[0])

    def test_file_scheme_rejected(self) -> None:
        self.assertIn("Unsupported link scheme", self.check("[Outside](file:///etc/passwd)").errors[0])

    def test_protocol_relative_rejected(self) -> None:
        self.assertEqual(len(self.check("[Other](//example.invalid/path)").errors), 1)

    def test_inventory_counted_separately(self) -> None:
        result = self.check("[Target](remote.md)", {"remote.md"})
        self.assertEqual(result.errors, [])
        self.assertEqual(result.inventory_links, 1)

    def test_inventory_does_not_imply_subtree(self) -> None:
        self.assertEqual(len(self.check("[Target](remote/unknown.md)", {"remote"}).errors), 1)

    def test_relative_parent_inside_repository(self) -> None:
        self.write("README.md")
        self.write("docs/start.md", "[Home](../README.md)")
        self.assertEqual(check_documents(self.root, ("docs/start.md",)).errors, [])

    def test_template_metadata(self) -> None:
        path = ".github/ISSUE_TEMPLATE/research.md"
        self.write(path, '---\nname: Research\nabout: A proposal\ntitle: "[research] "\nlabels: ""\nassignees: ""\n---\n')
        self.assertEqual(check_documents(self.root, (path,)).errors, [])

    def test_incomplete_template_rejected(self) -> None:
        path = ".github/ISSUE_TEMPLATE/research.md"
        self.write(path, "---\nname: Research\n---\n")
        self.assertEqual(len(check_documents(self.root, (path,)).errors), 1)

    def test_unclosed_metadata_rejected(self) -> None:
        self.assertEqual(issue_fields("---\nname: Research"), {})

    def test_duplicate_template_names_rejected(self) -> None:
        paths = (".github/ISSUE_TEMPLATE/a.md", ".github/ISSUE_TEMPLATE/b.md")
        for path in paths:
            self.write(path, "---\nname: Duplicate\nabout: Text\ntitle: Text\nlabels:\nassignees:\n---\n")
        self.assertEqual(len(check_documents(self.root, paths).errors), 1)

    def test_valid_inventory(self) -> None:
        path = self.write("inventory.json", json.dumps({
            "schema_version": 1, "source_commit": "a" * 40, "paths": ["remote.md"],
        }))
        self.assertEqual(load_inventory(path), {"remote.md"})

    def test_invalid_inventory(self) -> None:
        cases = (
            {"schema_version": 2, "source_commit": "a" * 40, "paths": []},
            {"schema_version": 1, "source_commit": "short", "paths": []},
            {"schema_version": 1, "source_commit": "a" * 40, "paths": ["../secret"]},
            {"schema_version": 1, "source_commit": "a" * 40, "paths": ["/secret"]},
            {"schema_version": 1, "source_commit": "a" * 40, "paths": [1]},
        )
        for case in cases:
            with self.subTest(case=case):
                path = self.write("inventory.json", json.dumps(case))
                with self.assertRaises(ValueError):
                    load_inventory(path)


if __name__ == "__main__":
    unittest.main()
