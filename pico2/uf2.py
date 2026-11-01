#!/usr/bin/env python3
"""ELF to UF2 for drag-and-drop flashing of an RP2350 (Arm, secure), no picotool needed.

Usage: python3 uf2.py target/thumbv8m.main-none-eabihf/release/tinystories-usb tinystories-usb.uf2

Writes what `picotool uf2 convert <elf> <uf2> --family rp2350-arm-s` writes for a
flash image: every loadable segment at its load address (LMA), in 256-byte pages,
gaps and page ends filled with zeros, one 512-byte UF2 block per page, in address
order. Check with --check against a UF2 made by another tool (blocks compared, and
with --ignore-family also when only the family ID differs).
"""
import struct
import sys

UF2_MAGIC_START0, UF2_MAGIC_START1, UF2_MAGIC_END = 0x0A324655, 0x9E5D5157, 0x0AB16F30
UF2_FLAG_FAMILY_ID_PRESENT = 0x00002000
RP2350_ARM_S = 0xE48BFF59
FLASH = range(0x10000000, 0x11000000)  # the 16 MiB XIP window
PAGE = 256


def segments(elf):
    if elf[:4] != b"\x7fELF" or elf[4] != 1 or elf[5] != 1:
        sys.exit("not a 32-bit little-endian ELF")
    machine, = struct.unpack_from("<H", elf, 18)
    if machine != 40:
        sys.exit("not an Arm ELF")
    phoff, = struct.unpack_from("<I", elf, 28)
    phentsize, phnum = struct.unpack_from("<HH", elf, 42)
    for i in range(phnum):
        p_type, p_offset, _vaddr, p_paddr, p_filesz = struct.unpack_from("<IIIII", elf, phoff + i * phentsize)
        if p_type == 1 and p_filesz:  # PT_LOAD with bytes in the file
            if p_paddr not in FLASH or p_paddr + p_filesz - 1 not in FLASH:
                sys.exit(f"segment at {p_paddr:#x} is not in flash")
            yield p_paddr, elf[p_offset:p_offset + p_filesz]


def pages(elf):
    out = {}
    for address, data in segments(elf):
        for i, byte in enumerate(data):
            page = out.setdefault((address + i) & ~(PAGE - 1), bytearray(PAGE))
            page[(address + i) % PAGE] = byte
    return sorted(out.items())


def uf2(elf, family=RP2350_ARM_S):
    blocks = pages(elf)
    return b"".join(
        struct.pack("<8I", UF2_MAGIC_START0, UF2_MAGIC_START1, UF2_FLAG_FAMILY_ID_PRESENT,
                    address, PAGE, n, len(blocks), family)
        + bytes(data) + bytes(476 - PAGE) + struct.pack("<I", UF2_MAGIC_END)
        for n, (address, data) in enumerate(blocks))


def main():
    args = sys.argv[1:]
    family = int(args[args.index("--family") + 1], 0) if "--family" in args else RP2350_ARM_S
    if "--check" in args:
        mine, theirs = uf2(open(args[0], "rb").read(), family), open(args[args.index("--check") + 1], "rb").read()
        counts = f"({len(mine) // 512} and {len(theirs) // 512} blocks)"
        if "--ignore-family" in args:
            mine, theirs = (b"".join(b[i:i + 28] + b[i + 32:i + 512] for i in range(0, len(b), 512)) for b in (mine, theirs))
        print("same" if mine == theirs else "DIFFERENT", counts)
        sys.exit(mine != theirs)
    data = uf2(open(args[0], "rb").read(), family)
    open(args[1], "wb").write(data)
    first, last = struct.unpack_from("<I", data, 12)[0], struct.unpack_from("<I", data, len(data) - 500)[0]
    print(f"{args[1]}: {len(data) // 512} blocks, {first:#x} to {last + PAGE:#x}, family {family:#x}")


if __name__ == "__main__":
    main()
