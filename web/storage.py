"""Необязательное видео сессии в SeaweedFS (S3)."""

from __future__ import annotations

import logging
import os
import time

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

log = logging.getLogger("shottracker.web")


def endpoint() -> str:
    return os.environ.get("S3_ENDPOINT", "http://seaweedfs:8333")


def bucket() -> str:
    return os.environ.get("S3_BUCKET", "shottracker")


def client():
    return boto3.client(
        "s3",
        endpoint_url=endpoint(),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY", "shottracker"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY", "shottracker-secret"),
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def ensure_bucket() -> bool:
    last: Exception | None = None
    name = bucket()
    for _ in range(30):
        try:
            s3 = client()
            try:
                s3.head_bucket(Bucket=name)
                return True
            except ClientError:
                s3.create_bucket(Bucket=name)
                return True
        except Exception as exc:
            last = exc
            time.sleep(1)
    log.warning("SeaweedFS недоступен, видео на сервер пока не сохранить: %s", last)
    return False


def put_bytes(key: str, data: bytes, content_type: str) -> None:
    client().put_object(Bucket=bucket(), Key=key, Body=data, ContentType=content_type)


def get_object(key: str):
    return client().get_object(Bucket=bucket(), Key=key)
