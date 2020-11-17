#! /usr/bin/env sh
set -e

exec celery -A app.core.celery_app worker --queues main-queue --loglevel=info --concurrency=1
