import re
from ai_memory.compression import MemoryCompressor
from ai_memory.token_counter import TokenCounter


def test_code_comments_removed():
    comp = MemoryCompressor()
    code = """
    def foo(): # inline comment
        pass  # TODO something

    # standalone comment
    print(foo())
    """
    out = comp.compress_code(code)
    assert "#" not in out
    assert "foo()" in out
    assert "pass" in out


def test_conversation_trimmed():
    comp = MemoryCompressor()
    long_text = " ".join([f"sentence{i}." for i in range(200)])
    out = comp.compress_text(long_text)
    tk = TokenCounter()
    assert tk.count(out) <= 120

