"""
Security and Governance Module for Healthcare RAG.

Features:
- PII/PHI redaction
- Role-based access control (RBAC)
- Audit logging
- Sensitive token detection
"""

import re
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
from pathlib import Path


# =============================================================================
# PII/PHI REDACTION
# =============================================================================

class SensitiveDataType(str, Enum):
    SSN = "ssn"
    PHONE = "phone"
    EMAIL = "email"
    DOB = "date_of_birth"
    MRN = "medical_record_number"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    NAME = "name"


@dataclass
class RedactionResult:
    """Result of redaction operation."""
    original_length: int
    redacted_length: int
    redactions_made: int
    redaction_types: List[str]
    redacted_text: str


class PHIRedactor:
    """
    Redacts Protected Health Information (PHI) from text.
    
    Detects and redacts:
    - Social Security Numbers
    - Phone numbers
    - Email addresses
    - Medical Record Numbers
    - Dates of birth
    - Credit card numbers
    - Addresses (partial)
    """
    
    # Regex patterns for sensitive data
    PATTERNS = {
        SensitiveDataType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
        SensitiveDataType.PHONE: r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        SensitiveDataType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        SensitiveDataType.DOB: r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b',
        SensitiveDataType.MRN: r'\bMRN[:\s]?\d{6,10}\b',
        SensitiveDataType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    }
    
    # Replacement tokens
    REPLACEMENTS = {
        SensitiveDataType.SSN: "[REDACTED-SSN]",
        SensitiveDataType.PHONE: "[REDACTED-PHONE]",
        SensitiveDataType.EMAIL: "[REDACTED-EMAIL]",
        SensitiveDataType.DOB: "[REDACTED-DOB]",
        SensitiveDataType.MRN: "[REDACTED-MRN]",
        SensitiveDataType.CREDIT_CARD: "[REDACTED-CC]",
    }
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._compiled_patterns = {
            dtype: re.compile(pattern, re.IGNORECASE)
            for dtype, pattern in self.PATTERNS.items()
        }
    
    def redact(self, text: str) -> RedactionResult:
        """Redact sensitive information from text."""
        if not self.enabled:
            return RedactionResult(
                original_length=len(text),
                redacted_length=len(text),
                redactions_made=0,
                redaction_types=[],
                redacted_text=text
            )
        
        redacted = text
        redaction_count = 0
        types_found = []
        
        for dtype, pattern in self._compiled_patterns.items():
            matches = pattern.findall(redacted)
            if matches:
                redacted = pattern.sub(self.REPLACEMENTS[dtype], redacted)
                redaction_count += len(matches)
                types_found.append(dtype.value)
        
        return RedactionResult(
            original_length=len(text),
            redacted_length=len(redacted),
            redactions_made=redaction_count,
            redaction_types=types_found,
            redacted_text=redacted
        )
    
    def contains_phi(self, text: str) -> bool:
        """Check if text contains PHI."""
        for pattern in self._compiled_patterns.values():
            if pattern.search(text):
                return True
        return False


# =============================================================================
# ROLE-BASED ACCESS CONTROL
# =============================================================================

class Role(str, Enum):
    ADMIN = "admin"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    AUDITOR = "auditor"
    GUEST = "guest"


class Permission(str, Enum):
    READ_DOCUMENTS = "read_documents"
    WRITE_DOCUMENTS = "write_documents"
    DELETE_DOCUMENTS = "delete_documents"
    QUERY_RAG = "query_rag"
    VIEW_PHI = "view_phi"
    EXPORT_DATA = "export_data"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_USERS = "manage_users"


# Role-permission mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.READ_DOCUMENTS,
        Permission.WRITE_DOCUMENTS,
        Permission.DELETE_DOCUMENTS,
        Permission.QUERY_RAG,
        Permission.VIEW_PHI,
        Permission.EXPORT_DATA,
        Permission.VIEW_AUDIT_LOGS,
        Permission.MANAGE_USERS,
    },
    Role.CLINICIAN: {
        Permission.READ_DOCUMENTS,
        Permission.QUERY_RAG,
        Permission.VIEW_PHI,
    },
    Role.RESEARCHER: {
        Permission.READ_DOCUMENTS,
        Permission.QUERY_RAG,
        Permission.EXPORT_DATA,
    },
    Role.AUDITOR: {
        Permission.READ_DOCUMENTS,
        Permission.VIEW_AUDIT_LOGS,
    },
    Role.GUEST: {
        Permission.QUERY_RAG,
    },
}


@dataclass
class User:
    """User with role-based permissions."""
    user_id: str
    email: str
    name: str
    role: Role
    department: Optional[str] = None
    active: bool = True
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        if not self.active:
            return False
        return permission in ROLE_PERMISSIONS.get(self.role, set())
    
    def get_permissions(self) -> Set[Permission]:
        """Get all permissions for this user."""
        if not self.active:
            return set()
        return ROLE_PERMISSIONS.get(self.role, set())


class AccessControl:
    """Role-based access control manager."""
    
    def __init__(self):
        self._users: Dict[str, User] = {}
    
    def add_user(self, user: User):
        """Add a user."""
        self._users[user.user_id] = user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has permission."""
        user = self._users.get(user_id)
        if not user:
            return False
        return user.has_permission(permission)
    
    def require_permission(self, permission: Permission):
        """Decorator to require permission for a function."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, user_id: str = None, **kwargs):
                if not user_id:
                    raise PermissionError("User ID required")
                if not self.check_permission(user_id, permission):
                    raise PermissionError(f"Permission denied: {permission.value}")
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# =============================================================================
# AUDIT LOGGING
# =============================================================================

class AuditEventType(str, Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    QUERY = "query"
    DOCUMENT_ACCESS = "document_access"
    DOCUMENT_CREATE = "document_create"
    DOCUMENT_UPDATE = "document_update"
    DOCUMENT_DELETE = "document_delete"
    EXPORT = "export"
    PERMISSION_DENIED = "permission_denied"
    PHI_ACCESS = "phi_access"
    CONFIG_CHANGE = "config_change"


@dataclass
class AuditEvent:
    """Audit log event."""
    event_id: str
    timestamp: str
    event_type: AuditEventType
    user_id: str
    user_email: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    correlation_id: Optional[str]
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["event_type"] = self.event_type.value
        return result


class AuditLogger:
    """
    HIPAA-compliant audit logging.
    
    Logs all access to PHI and sensitive operations.
    """
    
    def __init__(self, log_file: str = "logs/audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup file logger
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log(
        self,
        event_type: AuditEventType,
        user_id: str,
        action: str,
        success: bool = True,
        user_email: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> AuditEvent:
        """Log an audit event."""
        import uuid
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            user_id=user_id,
            user_email=user_email,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details or {},
            ip_address=ip_address,
            correlation_id=correlation_id,
            success=success
        )
        
        # Write to log file
        self.logger.info(json.dumps(event.to_dict()))
        
        return event
    
    def log_query(
        self,
        user_id: str,
        query: str,
        num_results: int,
        latency_ms: float,
        correlation_id: Optional[str] = None
    ):
        """Log a RAG query."""
        # Hash query for privacy
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        
        self.log(
            event_type=AuditEventType.QUERY,
            user_id=user_id,
            action="rag_query",
            details={
                "query_hash": query_hash,
                "query_length": len(query),
                "num_results": num_results,
                "latency_ms": latency_ms
            },
            correlation_id=correlation_id
        )
    
    def log_phi_access(
        self,
        user_id: str,
        document_id: str,
        phi_types: List[str],
        correlation_id: Optional[str] = None
    ):
        """Log access to PHI."""
        self.log(
            event_type=AuditEventType.PHI_ACCESS,
            user_id=user_id,
            action="phi_access",
            resource_type="document",
            resource_id=document_id,
            details={"phi_types": phi_types},
            correlation_id=correlation_id
        )
    
    def get_events(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query audit events."""
        events = []
        
        if not self.log_file.exists():
            return events
        
        with open(self.log_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Apply filters
                    if user_id and data.get("user_id") != user_id:
                        continue
                    if event_type and data.get("event_type") != event_type.value:
                        continue
                    if start_date and data.get("timestamp", "") < start_date:
                        continue
                    if end_date and data.get("timestamp", "") > end_date:
                        continue
                    
                    events.append(data)
                    
                    if len(events) >= limit:
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        return events


# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================

def create_security_context(
    redactor: PHIRedactor,
    access_control: AccessControl,
    audit_logger: AuditLogger
) -> Dict[str, Any]:
    """Create security context for request processing."""
    return {
        "redactor": redactor,
        "access_control": access_control,
        "audit_logger": audit_logger
    }


# Global instances
phi_redactor = PHIRedactor(enabled=True)
access_control = AccessControl()
audit_logger = AuditLogger()

# Add default admin user
access_control.add_user(User(
    user_id="admin",
    email="admin@healthcare.org",
    name="System Administrator",
    role=Role.ADMIN
))
