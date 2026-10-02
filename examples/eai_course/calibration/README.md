# Shared arm calibration — Windows and macOS

The single active calibration is maintained in the application repository:

- [Shared calibration and local-device contract](https://github.com/EAI-Course-2026/eai_project/blob/main/calibration/README.md)
- [Active joint parameters](https://github.com/EAI-Course-2026/eai_project/blob/main/calibration/scs215_so101.json)
- [Hardware verification evidence](https://github.com/EAI-Course-2026/eai_project/blob/main/calibration/scs215_so101.meta.json)
- [Current application setup](https://github.com/EAI-Course-2026/eai_project/blob/main/docs/course/README.md)

This fork supplies the LeRobot 0.6.2 framework dependency. Its course examples
and the [2026-09-28 snapshot](history/20260928_scs215_so101.reference.json) are
historical migration sources. That snapshot is not the current shared baseline;
it must not be copied into a teammate's HF cache as a current calibration.
Its original COM-based filename was a host label, not a physical robot identity.

All collaborators using the same unchanged arm share the active joint JSON.
Windows COM names and macOS device paths, camera indices and compute selection
belong in the application's ignored `configs/hardware.local.toml`. Never name
a shared calibration after a host port. Shared metadata records the hardware
asset, date, method and verified/unverified operations, without host paths.

Use the application repository's pinned uv installation and commands on both
platforms. The application continues to pin this framework at `6a07790`; this
documentation cleanup does not upgrade that dependency. Control and CUDA
training use separate locks. Changes to joints, IDs, assembly or EEPROM require
hardware revalidation rather than automatic recalibration when changing computers.
