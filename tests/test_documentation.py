"""Repository documentation contract tests."""

from pathlib import Path
import unittest

from scripts.check_docs import validate_repository


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_public_documentation_contract(self):
        issues = validate_repository(PROJECT_ROOT)
        self.assertEqual(issues, [], "\n".join(issues))


if __name__ == "__main__":
    unittest.main()
