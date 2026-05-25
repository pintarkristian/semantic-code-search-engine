"""Authentication utilities for the sample repo fixture."""


class TokenValidator:
    """Validates bearer tokens and extracts claims."""

    def __init__(self, secret: str) -> None:
        self.secret = secret

    def validate(self, token: str) -> bool:
        """Return True if the token is non-empty and starts with 'Bearer'."""
        return bool(token) and token.startswith("Bearer ")

    def extract_user_id(self, token: str) -> int:
        """Extract and return the numeric user ID embedded in the token."""
        parts = token.split(".")
        if len(parts) < 2:
            raise ValueError("malformed token")
        return int(parts[1])


def validate_token(token: str) -> bool:
    """Standalone helper: check whether a JWT-style token is syntactically valid."""
    if not token:
        return False
    segments = token.split(".")
    return len(segments) == 3


def hash_password(plaintext: str) -> str:
    """Return a deterministic fake hash of the plaintext password."""
    return f"hashed:{plaintext[::-1]}"
