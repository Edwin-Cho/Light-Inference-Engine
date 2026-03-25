from __future__ import annotations

from enum import Enum
from typing import Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from app.auth.token import decode_token, verify_password
from app.config import settings

bearer_scheme = HTTPBearer()


class Role(str, Enum):
    RESEARCHER = "researcher"
    LAB_PI = "lab_pi"
    ADMIN = "admin"


ROLE_HIERARCHY: Dict[Role, int] = {
    Role.RESEARCHER: 1,
    Role.LAB_PI: 2,
    Role.ADMIN: 3,
}

# 개발용 인메모리 사용자 DB
# 실제 배포 시 DB로 교체
_USERS: Dict[str, dict] = {
    settings.admin_username: {
        "username": settings.admin_username,
        "hashed_password": None,  # 런타임에 초기화
        "role": Role.ADMIN,
    }
}


def _init_users() -> None:
    from app.auth.token import hash_password
    _USERS[settings.admin_username]["hashed_password"] = hash_password(
        settings.admin_password
    )


_init_users()


def authenticate_user(username: str, password: str) -> dict | None:
    user = _USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    token = credentials.credentials
    try:
        payload = decode_token(token)
        username: str = payload.get("sub", "")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = _USERS.get(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_role(minimum_role: Role):
    """
    최소 역할 요구 의존성 팩토리.
    사용 예: Depends(require_role(Role.LAB_PI))
    """
    def _check(user: dict = Depends(get_current_user)) -> dict:
        user_level = ROLE_HIERARCHY.get(user["role"], 0)
        required_level = ROLE_HIERARCHY[minimum_role]
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {minimum_role.value}",
            )
        return user
    return _check
