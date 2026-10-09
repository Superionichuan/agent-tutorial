import unittest
from support import Scripted, reply, run


class Regression(unittest.TestCase):
    def test_zero_answer_is_preserved(self):
        answer, _ = run(Scripted(reply("done", {"answer": 0})))
        self.assertEqual(answer, str(0), f"explicit zero answer was replaced by {answer!r}")


if __name__ == "__main__":
    unittest.main()
