# Coal Mining Disaster switchpoint inference
Bayesian Change Point detection using Infer.NET

The model infers from data the likely switchpoint using disaster count per year alone.

The idea is to model the data sequence before a switchpoint using a Poisson random variable parametrized with \lambda_early. The data points after the switchpoint are modeled with another Poisson random variable but with potentially different \lambda_late.

The priors for both \lambda_early and \lambda_late is Gamma(2.0, 2.0)

The current model is implemented in Infer.NET 2.6 that has a bug and has been addressed with a workaround.
