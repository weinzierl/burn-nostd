/* QEMU mps2-an505 boots in secure mode with the vector table at 0x10000000.
 * RAM: secure alias of SSRAM2/3. Sized like the RP2350 (520 KiB) on purpose. */
MEMORY {
    FLASH : ORIGIN = 0x10000000, LENGTH = 4096K
    RAM   : ORIGIN = 0x38000000, LENGTH = 520K
}
