"""fuko's own agentic PR reviewer.

Unlike a single-shot harness that sends one compressed diff prompt to a model,
this reviewer gives the model a real checkout of the PR head plus read-only
code-navigation tools and asks it to *verify* findings by reading the
surrounding code before reporting them. Different context construction is the
point: same-pipeline reviewers converge on the same salient findings, so a
reviewer that walks the repo surfaces what diff-only passes miss.

The package is harness-agnostic by design: :mod:`sidecar.reviewer.prompt`
defines the review strategy and the structured output contract (the part that
is fuko's value), while :mod:`sidecar.reviewer.harness` adapts it to a concrete
agent runtime -- headless Claude Code first, with other runtimes able to slot in
behind the same ``run_review`` signature later. :mod:`sidecar.reviewer.checkout`
materializes the PR head and its diff. The fuko driver that plugs this into the
review pipeline is :mod:`sidecar.backends.agentic`.
"""
