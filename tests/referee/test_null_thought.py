import contextlib
import io
import unittest
from support import Scripted, reply, run


class Regression(unittest.TestCase):
    def test_console_accepts_or_rejects_null_thought_without_crashing(self):
        llm = Scripted(reply("done", {"answer": "ok"}, thought=None),
                       reply("done", {"answer": "ok"}))
        with contextlib.redirect_stdout(io.StringIO()):
            answer, _ = run(llm, verbose=True, on_event=None)
        self.assertEqual(answer, "ok")


if __name__ == "__main__":
    unittest.main()
