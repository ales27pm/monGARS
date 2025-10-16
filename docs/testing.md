# Testing Guidance

The monGARS test suite contains more than five hundred checks covering API
contracts, orchestration flows, and the embedding services added in the recent
backend work. Some of these suites (notably the chaos and integration tests)
can take several minutes to complete, so it is tempting to interrupt `pytest`
once local spot checks look healthy.

Do **not** stop the run early. Changes that touch shared configuration or the
embedding service can impact distant subsystems, and we have previously missed
regressions because the run was cancelled after a handful of fast tests.

To execute the full suite locally, run:

```bash
pytest
```

On the reference development container this takes roughly a minute and a half
(~90 seconds) and should complete with all 551 tests passing. If you need a
quiet log, you can continue to use the `make test` target which wraps the same
command with the `-q` flag.

Always capture the output in your change notes so reviewers can verify the
suite finished without manual intervention.
