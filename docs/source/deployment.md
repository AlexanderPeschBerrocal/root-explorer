# Deployment environments

**Current behaviour:** ROOT Explorer may be started locally with Python 3.12
or run from its published container image, as described in
[Installation](installation.md). In the current application, both ROOT file
and GDML input can use **Path** or **Upload**. A Path refers to the
filesystem accessible to the **server process**, not the browser user.

A Docker deployment can explicitly mount a read-only directory, then use
the corresponding **container path** in the UI. Browser uploads require
no filesystem mount. Plots are exported only as browser downloads, so no
deployment needs a writable output directory.

## Planned restricted deployments (not yet implemented)

The architecture tracked in
[issue #110](https://github.com/AlexanderPeschBerrocal/root-explorer/issues/110)
will add validated `local`, `internal`, and `public` runtime profiles
using the **same versioned Docker image**, with separate process/storage
instances and security controls:

- `local`: retain the current trusted single-user Path/Upload workflow.
- `internal`: RWTH-network access with only explicitly approved read-only
  filesystem roots. The question of network access vs authentication, and
  whether a common trusted directory is sufficient, must be resolved
  before offering this to general internal users.
- `public`: intended to permit browser uploads only; no arbitrary
  remote URLs or server-side filesystem paths.

**These profile restrictions are proposals, not current safeguards.**
Do not expose the unrestricted current service on the public internet
under the assumption that they are implemented. Infrastructure access
control and isolation also remain necessary.

The related [hosting issue #17](https://github.com/AlexanderPeschBerrocal/root-explorer/issues/17)
and [session storage issue #38](https://github.com/AlexanderPeschBerrocal/root-explorer/issues/38)
cover container deployment and storage limits, including the intended
three-file session cap. The active Streamlit per-file setting is 4,096 MiB;
see [Performance](performance.md).
