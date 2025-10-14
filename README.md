# Bayesian Change Point Detection (Coal Mining Disasters)

This repository implements Bayesian change point detection for the classic "coal mining disasters" time series using Microsoft's open-source probabilistic programming library (Infer.NET, published as `Microsoft.ML.Probabilistic`). The app estimates the most likely year where the Poisson rate of disasters changes.

The implementation lives in `BayesianChangePoint/Program.cs` and targets .NET Framework 4.6.1.

## Model

We assume a single change point that splits the series into an early and a late regime.

- Change point `k` is uniform over all indices
- Early rate `early_rate ~ Gamma(shape=2.0, rate=2.0)`
- Late rate `late_rate ~ Gamma(shape=2.0, rate=2.0)`
- For each year index `t`:
  - If `t < k`: `data[t] ~ Poisson(early_rate)`
  - Else: `data[t] ~ Poisson(late_rate)`

Notes:
- The dataset is the annual count of serious UK coal mining disasters (commonly used in Bayesian change point tutorials). The original series contains missing values marked as `-999`; in this implementation the provided array uses a cleaned version where missing values are replaced by `2`.
- The inference prints the posterior over the change point as a categorical (`Discrete`) distribution.

## How to build and run

This is a classic .NET Framework project using `packages.config`.

On Windows (recommended):
1. Open `BayesianChangePoint.sln` in Visual Studio 2019 or later.
2. Restore NuGet packages when prompted.
3. Build and run the `BayesianChangePoint` project.

On Linux/macOS:
- The project targets .NET Framework 4.6.1 and is easiest to build on Windows.
- If you have Mono installed, you can try:
  - Restore packages (e.g., with Visual Studio for Mac or `nuget restore` if available).
  - Build with `msbuild BayesianChangePoint.sln /p:Configuration=Release`.

When the app runs, it will print the posterior distribution over the switch point and wait for a key press (`Console.ReadKey()`).

## Interpreting output

- The output is a `Discrete` distribution over the index of the change point; larger mass around an index indicates a more likely switch.
- You can compute a point estimate (e.g., the maximum a posteriori index) or summarize with the mean of the posterior.

## Known issues and modernization opportunities

- The solution targets .NET Framework 4.6.1 and uses `packages.config`. Consider migrating to an SDK-style project (`.NET 8` or `.NET 6`) and `PackageReference` for better cross‑platform builds.
- Dependencies are pinned to an older `Microsoft.ML.Probabilistic` release (`0.3.1810.501`). Updating to the latest stable release may require minor API adjustments.
- The program blocks on `Console.ReadKey()` which is inconvenient in headless environments. You can remove that line for non-interactive runs.
- There are no automated tests; adding unit tests for the model structure and basic inference would improve maintainability.

## About PyMC

This repository is implemented in C# using Infer.NET, not Python/PyMC. If you need a Python version, the model maps directly to PyMC with the same generative story (Gamma-Poisson with a uniform change point). A separate PyMC implementation can be added alongside this project if desired.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` for details.
