#!/usr/bin/env python3
"""Compare a board's serial log with the stories QEMU wrote, and print the speeds.

Usage: python3 check_log.py board-log.txt ../out/qemu-tinystories-usb.txt

The log comes from tinystories-usb or tinystories-lcd (for example `just
board-log`). It is read from the last header ("... bytes of weights in flash")
on: each story ends with its line "N tokens in M ms ...". The QEMU file has the
same four stories, same prompts, temperatures and seeds, each ending with "N
tokens". Exit status 0 if every story in the log matches its QEMU story.
"""
import difflib
import re
import sys

BOARD_END = re.compile(r"^(\d+) tokens in (\d+) ms(?:, (\d+) ms of it in the model)?")
QEMU_END = re.compile(r"^(\d+) tokens$")


def stories(lines, end):
    """(text, match) for every story closed by an `end` line."""
    found, current = [], []
    for line in lines:
        m = end.match(line)
        if m:
            found.append(("\n".join(current).strip("\n"), m))
            current = []
        else:
            current.append(line.rstrip())
    return found


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    log = open(sys.argv[1], encoding="utf-8", errors="replace").read().replace("\r", "").split("\n")
    headers = [i for i, line in enumerate(log) if "bytes of weights in flash" in line]
    if not headers:
        sys.exit("no header in the log: was the port opened before the first story? Run it again.")
    board = stories(log[headers[-1] + 1:], BOARD_END)
    qemu = stories(open(sys.argv[2], encoding="utf-8").read().split("\n"), QEMU_END)
    print(f"{log[headers[-1] - 1].strip()}: {len(board)} complete stories after the last header")
    ok = bool(board)
    for n, ((text, m), (expected, e)) in enumerate(zip(board, qemu), 1):
        tokens, ms, model_ms = int(m[1]), int(m[2]), int(m[3] or m[2])
        same = text == expected and tokens == int(e[1])
        speed = f"{tokens} tokens in {ms} ms" + (f" ({model_ms} ms in the model)" if m[3] else "")
        print(f"story {n}: {'same text as QEMU' if same else 'DIFFERENT from QEMU'}; {speed}: {tokens * 1000 / model_ms:.1f} tokens/s")
        if not same:
            ok = False
            for line in difflib.unified_diff(expected.split("\n"), text.split("\n"), "qemu", "board", lineterm="", n=1):
                print("    " + line)
    for n in range(len(qemu) + 1, len(board) + 1):
        tokens, ms, model_ms = int(board[n - 1][1][1]), int(board[n - 1][1][2]), int(board[n - 1][1][3] or board[n - 1][1][2])
        print(f"story {n}: no QEMU story to compare; {tokens} tokens in {ms} ms: {tokens * 1000 / model_ms:.1f} tokens/s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
