# Bayesian Change Point Detection (Coal Mining Disasters)

This repository implements Bayesian change point detection for the classic "coal mining disasters" time series using Microsoft's open-source probabilistic programming library (Infer.NET, published as `Microsoft.ML.Probabilistic`). The app estimates the most likely year where the Poisson rate of disasters changes.

The implementation lives in `BayesianChangePoint/Program.cs` and targets .NET 8 with an SDK-style project.

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

Requires .NET SDK 8.0+.

Using the dotnet CLI:
1. Restore and build
   
   ```bash
   dotnet build ./BayesianChangePoint/BayesianChangePoint.csproj -c Release
   ```

2. Run
   
   ```bash
   dotnet run --project ./BayesianChangePoint/BayesianChangePoint.csproj -c Release
   ```

When the app runs, it will print the posterior distribution over the switch point.

## Interpreting output

- The output is a `Discrete` distribution over the index of the change point; larger mass around an index indicates a more likely switch.
- You can compute a point estimate (e.g., the maximum a posteriori index) or summarize with the mean of the posterior.

## Notes and potential improvements

- Dependencies use the latest available `Microsoft.ML.Probabilistic` packages via `PackageReference`.
- The program previously blocked on `Console.ReadKey()`; it has been removed for headless runs.
- There are no automated tests; adding unit tests for the model structure and basic inference would improve maintainability.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` for details.
