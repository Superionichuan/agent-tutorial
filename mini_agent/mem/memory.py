"""Memory management"""
from pathlib import Path
from datetime import datetime


class Memory:
    """Memory manager"""

    def __init__(self, mem_dir: Path = None):
        if mem_dir is None:
            # defaults to mem/ at the project root
            mem_dir = Path(__file__).parent.parent.parent / "mem"
        self.mem_dir = Path(mem_dir)
        self.mem_dir.mkdir(exist_ok=True)

    @property
    def long_term_file(self) -> Path:
        return self.mem_dir / "long_term.md"

    @property
    def today_log_file(self) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        return self.mem_dir / f"{today}.md"

    def get_long_term(self, budget_chars: int = 3000) -> str:
        """Read long-term memory (with an insertion budget: keep head + recent tail when over budget)."""
        if not self.long_term_file.exists():
            return ""
        text = self.long_term_file.read_text()
        if len(text) <= budget_chars:
            return text
        head, tail_n = budget_chars // 3, budget_chars * 2 // 3
        return (text[:head] + "\n...[middle section omitted; consider distilling]...\n" + text[-tail_n:])

    def distill(self, llm) -> bool:
        """Distill: LLM merges and rewrites long-term memory (original backed up as .bak). Long-run maintenance."""
        if not self.long_term_file.exists():
            return False
        text = self.long_term_file.read_text()
        distilled = llm(
            "Distill and rewrite the following long-term memory notes: merge duplicates / keep every unique fact and preference / group by topic / be concise:\n"
            + text + "\n\nDistilled memory:")
        if not distilled or len(distilled) >= len(text):
            return False                       # convergence guard
        self.long_term_file.with_suffix(".md.bak").write_text(text)
        self.long_term_file.write_text(distilled + "\n")
        return True

    def get_today_log(self) -> str:
        """Read today's log"""
        if self.today_log_file.exists():
            return self.today_log_file.read_text()
        return ""

    def append_log(self, content: str):
        """Append to today's log"""
        with open(self.today_log_file, "a") as f:
            f.write(f"\n{content}\n")

    def append_memory(self, content: str):
        """Append to long-term memory"""
        with open(self.long_term_file, "a") as f:
            f.write(f"\n{content}\n")

    def save_output(self, filename: str, content: str):
        """Save generated code under output/"""
        output_dir = self.mem_dir.parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / filename
        output_file.write_text(content)
        return str(output_file)


# convenience functions
_default_mem = None


def _get_mem() -> Memory:
    global _default_mem
    if _default_mem is None:
        _default_mem = Memory()
    return _default_mem


def get_long_term() -> str:
    return _get_mem().get_long_term()


def get_today_log() -> str:
    return _get_mem().get_today_log()


def append_log(content: str):
    _get_mem().append_log(content)


def append_memory(content: str):
    _get_mem().append_memory(content)
