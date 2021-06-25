Machine learning (ML) models trained on personal data have been shown to leak information about users. Differential privacy (DP) enables model training with a guaranteed bound on this leakage. Each new model trained with DP increases the bound on data leakage and can be seen as consuming part of a global privacy budget that should not be exceeded. This
budget is a *scarce resource* that must be carefully managed to maximize the number of successfully trained models.

PrivateKube is an extension to the popular Kubernetes datacenter orchestrator that adds privacy as a new type of resource to be managed alongside other traditional compute resources, such as CPU, GPU, and memory.  The abstractions we design for the privacy resource mirror those defined by Kubernetes for traditional resources, but there are also major differences. For example, traditional compute resources are replenishable while privacy is not: a CPU can be regained after a model finishes execution while privacy
budget cannot. This distinction forces a re-design of the scheduler.  We developed *Dominant Private Block Fairness (DPF)* -- a variant of the popular *Dominant Resource Fairness (DRF)* algorithm -- that is geared toward the non-replenishable privacy resource but enjoys similar theoretical properties as DRF.

The design, implementation, and evaluation of PrivateKube and DPF are described in a paper published at OSDI 2021: [Privacy Budget Scheduling](https://www.usenix.org/conference/osdi21/presentation/luo).  A local copy of this paper is available [here](https://columbia.github.io/PrivateKube/papers/osdi2021privatekube.pdf).  An extended version of this paper, with some details we omitted from the conference paper, will be made available shortly.

The [PrivateKube repository](https://github.com/columbia/PrivateKube) contains the code we release as a reusable and extensible artifact of our research.

## Related Publications

* Tao Luo, Mingen Pan, Pierre Tholoniat, Asaf Cidon, Roxana Geambasu, and Mathias Lecuyer. "Privacy Budget Scheduling and Orchestration." In *Proceedings of the USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, July 2021. [PDF](papers/osdi2021privatekube.pdf)

* Mathias Lecuyer, Riley Spahn, Kiran Vodrahalli, Roxana Geambasu, and Daniel Hsu. "Privacy Accounting and Quality Control in the Sage Differentially Private Machine Learning Platform." In *Proceedings of the ACM Symposium on Operating Systems Principles (SOSP)*, October 2019. [PDF](papers/sosp2019sage.pdf)
