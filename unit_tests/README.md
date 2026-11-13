# Unit tests

Keep tests that can catch an incorrect result, broken public contract, resource
lifetime bug, or a previously observed failure. Test the production entry point
with small inputs; mock external dependencies only as needed.

- Add a regression case to an existing test when it exercises the same behavior.
  Parameterize inputs instead of copying setup and assertions into another test.
- For kernels, compare outputs and mutated state against an independent reference.
  Include relevant dtype, layout, padding, and boundary cases without taking the
  Cartesian product of unrelated options.
- Do not test class inheritance, private method inventories, or dataclass field
  declarations. Exercise the behavior those implementations support.
- A CPU reimplementation tested only against its own expected output does not
  validate the CUDA path. Run the real implementation or explicitly skip when its
  prerequisites are unavailable.
- Keep timing loops and performance reports out of unit tests. Benchmarks and
  service-level tests belong under `test/`; ordinary unit tests need assertions.
- When removing a test, check whether another test covers the same failure mode.
  Similar setup alone does not make cache ownership, transfer boundaries, or
  asynchronous failure tests redundant.

Run affected tests with `python -m pytest <paths>` in an environment with their
required dependencies. For changed Python files, run
`pre-commit run --files <paths>`. Report skipped hardware/provider checks separately
from passing tests.
