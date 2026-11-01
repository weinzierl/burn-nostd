#!/usr/bin/env python3
"""Show and save what a board running tinystories-usb or tinystories-lcd writes
over USB serial.

Usage: python3 board_log.py [PORT] [OUT] [STORIES]
       (defaults: /dev/ttyACM0 board-log.txt 4; on macOS the port is /dev/cu.usbmodem...)

Opening the port starts a session, and the board begins again with story 1
(tinystories-lcd after the story it is writing). The text is saved as it
arrives; the program stops after STORIES stories counted from the session's
header (so the end of an earlier story does not count), or on Ctrl-C.
"""
import errno
import os
import re
import sys
import tty

STORY_END = re.compile(rb"^\d+ tokens in \d+ ms")
HEADER = re.compile(rb"bytes of weights in flash")


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    out = sys.argv[2] if len(sys.argv) > 2 else "board-log.txt"
    stories = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    try:
        fd = os.open(port, os.O_RDWR | os.O_NOCTTY)
    except OSError as e:
        hint = "; on Linux: sudo usermod -aG dialout $(id -un), then log in again" if e.errno == errno.EACCES else ""
        sys.exit(f"cannot open {port}: {e.strerror}{hint}")
    tty.setraw(fd)  # no echo, no line-end translation
    done, partial, started = 0, b"", False
    with open(out, "wb") as f:
        try:
            while done < stories:
                data = os.read(fd, 1024)
                if not data:
                    break
                f.write(data)
                f.flush()
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
                *lines, partial = (partial + data).split(b"\n")
                for line in lines:
                    if HEADER.search(line):
                        started, done = True, 0
                    elif started and STORY_END.match(line):
                        done += 1
        except KeyboardInterrupt:
            pass
    print(f"\nsaved {out}: {done} complete stories")


if __name__ == "__main__":
    main()
