# ruff: noqa: UP042
from __future__ import annotations

from enum import Enum


class PlanSection(str, Enum):
    """12 required sections in the plan output format."""
    OVERVIEW = "Overview"
    CRITICAL_FILES = "Critical Files"
    ARCHITECTURE = "Architecture & Design"
    IMPLEMENTATION_STEPS = "Implementation Steps"
    FILES_TO_MODIFY = "Files to Create/Modify"
    DATABASE_CHANGES = "Database Changes"
    TESTING_STRATEGY = "Testing Strategy"
    VERIFICATION = "Verification Checklist"
    RISKS = "Risks & Edge Cases"
    DEPENDENCIES = "Dependencies & Prerequisites"
    OPEN_QUESTIONS = "Open Questions"
    DOCUMENTATION = "Documentation References"


class AuditCategory(str, Enum):
    """10 audit categories from build_audit_prompt."""
    MISSING_STEPS = "MISSING STEPS"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    EDGE_CASES = "EDGE CASES"
    CONVENTION_VIOLATIONS = "CONVENTION VIOLATIONS"
    TESTING_GAPS = "TESTING GAPS"
    BETTER_ALTERNATIVES = "BETTER ALTERNATIVES"
    DEPENDENCY_RISKS = "DEPENDENCY RISKS"
    ORDERING = "ORDERING"
    COMPLETENESS = "COMPLETENESS"


class AuditSeverity(str, Enum):
    """Severity levels for audit findings."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SUGGESTION = "SUGGESTION"


class AuditVerdict(str, Enum):
    """Overall audit verdict."""
    APPROVE = "APPROVE"
    NEEDS_REVISION = "NEEDS_REVISION"
    MAJOR_CONCERNS = "MAJOR_CONCERNS"


class FindingVerdict(str, Enum):
    """Per-finding verdict in refinement appendix."""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class PlanPhase(str, Enum):
    """4 phases in the planning workflow prompt."""
    EXPLORE = "EXPLORE"
    ANALYZE = "ANALYZE"
    DESIGN = "DESIGN"
    PLAN = "PLAN"


class Complexity(str, Enum):
    """Step complexity tags."""
    SMALL = "S"
    MEDIUM = "M"
    LARGE = "L"
