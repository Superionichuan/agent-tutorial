import unittest
import support
from mini_agent.tools.math.calc import calc


class Regression(unittest.TestCase):
    def test_nonmathematical_literal_is_rejected(self):
        result = calc("...")
        self.assertTrue(result.startswith("Error:"), f"non-numeric expression accepted: {result!r}")


if __name__ == "__main__":
    unittest.main()
