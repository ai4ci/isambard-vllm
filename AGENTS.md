# Principles

* If the user prompt looks like a bash command they have probably forgotten to escape it and you should stop and ask whether they forgot to escape it.
* If the input looks incomplete ask the user and wait for the next message.
* If uncertain about user intentions clarify with the user and do not make assumptions.
* Design artefacts are in the `design` folder.
* Design requirements (`design/requirements.md`) must be discussed and evolved in discussion with the user.
* Confirm and review requirements critically with a YAGNI approach.
* Make clarifications to requirements and identify minimal viable product.
* Until roadmap is signed off assume the requirements may be updated by the user and will need to be frequently rechecked.
* Assume all scripts in the `design` folder are instructional examples and must be completely rewritten.
* Document architecture decisions and a phased roadmap (as a checklist) before implementation (`design/adr.md`, `design/roadmap.md`).
* Focus on minimal viable product before implementing additional features.
* Get sign off for the roadmap from the user.
* For each roadmap phase generate an implementation checklist before writing code.
* Commit changes before each roadmap phase.
* It is **mandatory** to implement using test driven development.
* All tests must be implemented and proven to fail before starting development of a feature.
* All tests must be proven to pass before development of a feature is complete.
* Bump minor version numbers with every commit.
* After development of a roadmap phase update the design roadmap and implementation checklists and seek confirmation from user before proceeding to next roadmap phase.
* If uncertain about anything ask the user, do not make assumptions.
