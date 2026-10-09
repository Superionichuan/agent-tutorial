import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from support import skill
from mini_agent.skills import loader


class Regression(unittest.TestCase):
    def test_invalid_skill_entry_does_not_break_valid_skills(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill(root)
            (root / "broken" / "SKILL.md").mkdir(parents=True)
            with patch.object(loader, "SKILL_DIRS", [root]):
                loaded = loader.load_skills()
        self.assertIn("demo", loaded)
        self.assertNotIn("broken", loaded)


if __name__ == "__main__":
    unittest.main()
