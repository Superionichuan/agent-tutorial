import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from support import skill
from mini_agent.skills import loader
from mini_agent.tools.skill.read_skill import read_skill


class Regression(unittest.TestCase):
    def test_full_instructions_preserve_indented_code_block(self):
        body = "    print('first')\n    print('second')"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill(root, body)
            with patch.object(loader, "SKILL_DIRS", [root]), patch.object(loader, "_skills_cache", None):
                actual = read_skill("demo")
        self.assertEqual(actual, body, f"instruction indentation changed to {actual!r}")


if __name__ == "__main__":
    unittest.main()
