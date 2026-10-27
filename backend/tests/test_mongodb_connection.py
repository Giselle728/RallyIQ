
from unittest.mock import MagicMock, patch

from pymongo.errors import ServerSelectionTimeoutError

from backend.storage.mongo_store import (
    MongoStorage,
    safe_mongo_error,
)


def test_timeout_error_is_safe():
    """Ensure MongoDB timeout errors don't expose sensitive details."""
    error = ServerSelectionTimeoutError(
        "mongodb://user:secret@private-host timed out"
    )

    message = safe_mongo_error(error)

    assert "timed out" in message.lower()
    assert "secret" not in message
    assert "private-host" not in message


def test_unexpected_error_is_sanitized():
    """Ensure unexpected errors don't expose credentials or hostnames."""
    error = RuntimeError(
        "mongodb://admin:secret@private-host failed"
    )

    message = safe_mongo_error(error)

    assert "secret" not in message
    assert "private-host" not in message


@patch("backend.storage.mongo_store.mongomock.MongoClient")
@patch("backend.storage.mongo_store.pymongo.MongoClient")
def test_connection_failure_activates_fallback(
    mock_mongo_client,
    mock_mongomock_client,
):
    """Ensure a failed live connection closes its client and uses fallback."""
    # Simulate a failed live MongoDB connection.
    failed_client = MagicMock()
    failed_client.admin.command.side_effect = (
        ServerSelectionTimeoutError("connection timed out")
    )
    mock_mongo_client.return_value = failed_client

    # Simulate the embedded mongomock fallback.
    fallback_client = MagicMock()
    mock_mongomock_client.return_value = fallback_client

    # Create a fresh instance without changing the shared singleton.
    storage = object.__new__(MongoStorage)
    storage._initialized = False
    storage.uri = "mongodb://localhost:27017"
    storage.db_name = "test_db"
    storage.is_atlas = False
    storage.target_cluster = "localhost:27017"
    storage.is_live_mongo = False
    storage.connection_status = "unconnected"
    storage.connection_error = None
    storage.client = None
    storage.db = None

    # Run the connection logic.
    storage._connect()

    # Verify cleanup and fallback behavior.
    failed_client.close.assert_called_once()
    mock_mongomock_client.assert_called_once()

    assert storage.is_live_mongo is False
    assert storage.connection_status == "fallback_mock"
    assert storage.connection_error is not None
    assert storage.client is fallback_client
