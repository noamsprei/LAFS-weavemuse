"""Trace-quality evaluation harness for WeaveMuse.

Additive, self-contained package: runs the manager `CodeAgent` over a dataset
of tasks under different prompt ("instructions") variants, captures the full
step-by-step execution trace for each (task, variant) pair, and scores those
traces with an LLM judge. See weavemuse/eval/README.md for the full guide.

Nothing in this package is imported by, or modifies, the rest of WeaveMuse.
"""
