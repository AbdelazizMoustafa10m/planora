from __future__ import annotations

from enum import StrEnum


class PlanSection(StrEnum):
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


class AuditCategory(StrEnum):
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


class AuditSeverity(StrEnum):
    """Severity levels for audit findings."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SUGGESTION = "SUGGESTION"


class AuditVerdict(StrEnum):
    """Overall audit verdict."""

    APPROVE = "APPROVE"
    NEEDS_REVISION = "NEEDS_REVISION"
    MAJOR_CONCERNS = "MAJOR_CONCERNS"


class FindingVerdict(StrEnum):
    """Per-finding verdict in refinement appendix."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class PlanPhase(StrEnum):
    """4 phases in the planning workflow prompt."""

    EXPLORE = "EXPLORE"
    ANALYZE = "ANALYZE"
    DESIGN = "DESIGN"
    PLAN = "PLAN"


class Complexity(StrEnum):
    """Step complexity tags."""

    SMALL = "S"
    MEDIUM = "M"
    LARGE = "L"
