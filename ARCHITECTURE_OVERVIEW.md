# ARCHITECTURE_OVERVIEW

**Document:** Architecture overview  
**Version:** 1.0.0  
**Date:** 6 November 2026  
**Status:** Architecture reference — documentary  
**Scope:** `eden-foot-models` and `eden-engine`

---

## 1. Purpose

This document records the high-level separation of responsibilities between the two repositories. It is intended to provide a common architectural reference without creating an implicit runtime dependency between them.

It describes repository boundaries, governance expectations, documentation practices, and criteria for reviewing architectural changes. It does not certify the current implementation, repository configuration, security posture, or CI status.

## 2. Architectural principles

The architecture is guided by the following principles:

- **Separation of concerns:** each repository has a distinct responsibility and lifecycle.
- **Explicit boundaries:** dependencies and data exchanges must be documented rather than assumed.
- **Least privilege:** access rights should be limited to what each component requires.
- **Reproducibility:** decisions and validation evidence should be traceable.
- **Fail-safe changes:** changes should be reviewed and validated before integration.
- **Independent governance:** each repository maintains its own history, reviews, and CI.

## 3. Repository landscape

```text
+------------------------------+       +------------------------------+
| eden-foot-models              |       | eden-engine                  |
|------------------------------|       |------------------------------|
| Research and analysis         |       | Application and infrastructure|
| Method documentation          |       | Operational controls          |
| Independent validation        |       | Persistence and interfaces    |
+------------------------------+       +------------------------------+
              |                                         |
              +---------- documentary coordination ----+
                         No implicit runtime coupling
```

The diagram expresses responsibility boundaries, not a verified deployment topology.

## 4. Repository responsibilities

### 4.1 `eden-foot-models`

This repository is the location for research-oriented material and the documentation of analytical methods.

Its architectural responsibilities are limited to:

- documenting the purpose and assumptions of research artifacts;
- recording datasets' provenance and known limitations;
- describing validation methodology and reproducibility requirements;
- preserving research results with clear version and evidence references;
- distinguishing exploratory work from reviewed, stable material.

Research artifacts must not be treated as production services merely because they exist in this repository.

### 4.2 `eden-engine`

This repository is the location for application and infrastructure concerns.

Its architectural responsibilities include:

- defining application boundaries and interfaces;
- documenting persistence and transaction responsibilities;
- recording operational, access-control, and security decisions;
- describing deployment and maintenance responsibilities;
- maintaining independent quality gates and change history.

Operational behavior must be specified and reviewed within this repository rather than inferred from research documentation.

## 5. Boundary and interaction rules

The repositories remain independent unless a separately reviewed architecture decision explicitly establishes an interface.

Any future exchange must document, at minimum:

1. the purpose and owner of the interface;
2. the format and meaning of exchanged information;
3. versioning and compatibility expectations;
4. validation and error-handling responsibilities;
5. access-control and data-governance requirements;
6. evidence required before the interface is accepted.

A documentation reference does not itself authorize a runtime dependency, data transfer, or deployment coupling.

## 6. Repository governance

Each repository should maintain its own:

- protected default branch;
- focused feature or maintenance branches;
- pull-request review process;
- automated checks appropriate to that repository;
- release and change history.

Changes should be integrated through the repository's normal review process. Direct changes to a protected default branch should be avoided unless an explicitly approved repository policy allows them.

## 7. Documentation and evidence

Architectural decisions should be recorded in a consistent format:

- **Context:** the problem or constraint;
- **Decision:** the selected approach;
- **Alternatives:** relevant options considered;
- **Consequences:** trade-offs and follow-up work;
- **Evidence:** links to reviews, tests, or other supporting records.

Validation statements must identify what was actually checked, where it was checked, and the relevant revision. A passing check in one repository does not imply that the other repository was checked.

## 8. Change review checklist

Before accepting an architectural change, reviewers should confirm:

- [ ] The change belongs to the repository being modified.
- [ ] The change preserves the documented repository boundary.
- [ ] Any new dependency or interface is explicitly documented.
- [ ] Security and access implications have been considered.
- [ ] Relevant tests and checks are identified and reported accurately.
- [ ] Documentation reflects the behavior actually implemented.
- [ ] No unverified deployment, certification, or compatibility claim is made.

## 9. Known limitations

This overview is a governance and documentation artifact. It does not assert that:

- either repository has been inspected in full;
- the described responsibilities match every current source file;
- a CI pipeline has passed;
- a deployment exists or is operational;
- any cryptographic, mathematical, or security certification has been performed.

Those claims require separate, current evidence.

## 10. Acceptance criteria for this document

This document is suitable for archival when:

1. the same reviewed version is present at the root of each intended repository;
2. the repository maintainers confirm that the responsibility descriptions are accurate;
3. repository-specific implementation details are documented in the relevant repository;
4. any divergence between the two copies is handled through an explicit review.

---

**End of document**
