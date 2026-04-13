"""Training entrypoints package.

Keep package initialization side-effect free so individual entrypoints can be
executed via ``python -m methods.<name>`` without importing unrelated modules.
"""
