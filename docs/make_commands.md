## Makefile Compatibility

A `Makefile` is provided as a thin compatibility layer for the
[gdsfactory PDK CI workflows](https://github.com/doplaydo/pdk-ci-workflow). **`just` remains the primary task runner**
for day-to-day development; the Makefile exists so that `make install` and `make test` work out-of-the-box in CI
environments that expect them.

```bash
make docs      # equivalent to: just docs
make build     # equivalent to: just build
make run-pre   # equivalent to: just run-pre
```
