"""
logger/__init__.py
Makes `logger` a proper Python package.
"""
from logger.audit_logger import _audit

__all__ = ["_audit"]
