import unittest
from support import Scripted, reply, run


class Regression(unittest.TestCase):
    def test_budget_stops_retry_after_invalid_reply(self):
        llm = Scripted("no JSON", reply("done", {"answer": "overspent"}))
        _, events = run(llm, max_tokens=1)
        reason = [data["reason"] for kind, data in events if kind == "turn/end"]
        self.assertEqual((llm.calls, reason), (1, ["token_limit"]),
                         f"calls={llm.calls}, end reasons={reason!r}")


if __name__ == "__main__":
    unittest.main()
