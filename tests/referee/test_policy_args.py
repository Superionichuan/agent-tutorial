import unittest
from support import Scripted, reply, run


class Regression(unittest.TestCase):
    def test_malformed_tool_args_do_not_crash_dictionary_policy(self):
        invoked = []
        llm = Scripted(reply("probe", []), reply("done", {"answer": "ok"}))
        answer, _ = run(
            llm, tools={"probe": lambda **kw: invoked.append(kw)},
            approve=lambda action, args: args.get("allowed", False))
        self.assertEqual(invoked, [])
        self.assertEqual(answer, "ok")


if __name__ == "__main__":
    unittest.main()
