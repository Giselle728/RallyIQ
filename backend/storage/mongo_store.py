
"""
RallyIQ MongoDB Storage Layer
Provides document storage, indexing, and querying for matches, points, player profiles,
and streaming events. Supports live MongoDB with automatic fallback to high-performance
embedded mongomock with JSON disk persistence.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

import certifi
import pymongo
import mongomock
from pymongo.errors import (
    ConnectionFailure,
    ServerSelectionTimeoutError,
    OperationFailure,
)

from backend.config import MONGO_URI, MONGO_DB_NAME, MONGO_DISK_CACHE


logger = logging.getLogger("rallyiq.mongo")


def mask_mongo_uri(uri: str) -> str:
    """Mask credentials in MongoDB connection string for safe logging."""
    if not uri:
        return ""

    try:
        parsed = urlparse(uri)

        if parsed.username or parsed.password:
            netloc = parsed.netloc

            if "@" in netloc:
                _, host = netloc.split("@", 1)
                masked_netloc = "****:****@" + host
                return uri.replace(netloc, masked_netloc)

        return uri

    except Exception:
        return "mongodb+srv://****:****@<masked-cluster>"


def safe_mongo_error(error: Exception) -> str:
    """Return a safe MongoDB error message without exposing secrets."""
    if isinstance(error, ServerSelectionTimeoutError):
        return "MongoDB server is unavailable or the connection timed out."

    if isinstance(error, ConnectionFailure):
        return "Could not establish or maintain a MongoDB connection."

    if isinstance(error, OperationFailure):
        return "MongoDB rejected the operation. Check database permissions."

    return "MongoDB connection failed due to an unexpected error."


class MongoStorage:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MongoStorage, cls).__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(
        self,
        uri: str = MONGO_URI,
        db_name: str = MONGO_DB_NAME,
    ):
        if self._initialized:
            return

        self.uri = uri
        self.db_name = db_name
        self.is_live_mongo = False
        self.is_atlas = "mongodb+srv" in uri or "mongodb.net" in uri
        self.target_cluster = "localhost:27017"
        self.connection_status = "unconnected"
        self.connection_error = None
        self.client = None
        self.db = None

        if self.uri:
            try:
                parsed = urlparse(self.uri)

                if parsed.hostname:
                    self.target_cluster = parsed.hostname

            except Exception:
                pass

        self._connect()
        self._setup_indexes()
        self._load_from_disk()
        self._initialized = True

    def _connect(self):
        """Connect to MongoDB or safely fall back to embedded storage."""
        test_client = None

        try:
            connect_kwargs: Dict[str, Any] = {
                "serverSelectionTimeoutMS": 4000,
                "connectTimeoutMS": 4000,
                "socketTimeoutMS": 4000,
            }

            # Keep TLS certificate verification enabled.
            if (
                self.is_atlas
                or "ssl=true" in self.uri.lower()
                or "tls=true" in self.uri.lower()
            ):
                connect_kwargs["tlsCAFile"] = certifi.where()

            test_client = pymongo.MongoClient(
                self.uri,
                **connect_kwargs,
            )

            # Verify connectivity before selecting the live client.
            test_client.admin.command("ping")

            self.client = test_client
            self.db = self.client[self.db_name]
            self.is_live_mongo = True
            self.connection_status = (
                "connected_atlas"
                if self.is_atlas
                else "connected_local"
            )
            self.connection_error = None

            logger.info(
                "Connected to MongoDB (%s, database: %s)",
                self.target_cluster,
                self.db_name,
            )

        except Exception as exc:
            # Close the unsuccessful client without hiding the original failure.
            if test_client is not None:
                try:
                    test_client.close()
                except Exception:
                    logger.debug(
                        "Failed to close unsuccessful MongoDB client."
                    )

            self.connection_error = safe_mongo_error(exc)
            self.connection_status = "fallback_mock"
            self.is_live_mongo = False

            logger.warning(
                "MongoDB connection unavailable (%s). "
                "Using mongomock with disk persistence.",
                type(exc).__name__,
            )

            # Preserve the existing embedded storage fallback.
            self.client = mongomock.MongoClient()
            self.db = self.client[self.db_name]

    def _setup_indexes(self):
        """Create useful indexes on collections."""
        try:
            self.db.matches.create_index(
                [("match_id", pymongo.ASCENDING)],
                unique=True,
            )
            self.db.matches.create_index(
                [("Player 1", pymongo.ASCENDING)]
            )
            self.db.matches.create_index(
                [("Player 2", pymongo.ASCENDING)]
            )
            self.db.matches.create_index(
                [("Surface", pymongo.ASCENDING)]
            )
            self.db.matches.create_index(
                [("Year", pymongo.ASCENDING)]
            )

            self.db.points.create_index(
                [
                    ("match_id", pymongo.ASCENDING),
                    ("Pt", pymongo.ASCENDING),
                ]
            )

            self.db.player_profiles.create_index(
                [("player", pymongo.ASCENDING)],
                unique=True,
            )

            logger.info("MongoDB indexes successfully created.")

        except Exception as exc:
            logger.error(
                "Could not create MongoDB indexes (%s).",
                type(exc).__name__,
            )

    def _load_from_disk(self):
        """Load cached documents from disk if using embedded mongomock."""
        if not self.is_live_mongo and os.path.exists(MONGO_DISK_CACHE):
            try:
                with open(MONGO_DISK_CACHE, "r", encoding="utf-8") as file:
                    data = json.load(file)

                    for col_name, docs in data.items():
                        if docs:
                            self.db[col_name].drop()
                            self.db[col_name].insert_many(docs)

                logger.info(
                    "Loaded existing documents from disk cache."
                )

            except Exception as exc:
                logger.warning(
                    "Could not load disk cache (%s).",
                    type(exc).__name__,
                )

    def persist_to_disk(self):
        """Persist current collections to disk."""
        if not self.is_live_mongo:
            try:
                data = {}

                for col_name in ["matches", "player_profiles"]:
                    cursor = self.db[col_name].find()
                    docs = []

                    for doc in cursor:
                        doc_copy = dict(doc)

                        if "_id" in doc_copy:
                            doc_copy["_id"] = str(doc_copy["_id"])

                        docs.append(doc_copy)

                    data[col_name] = docs

                with open(
                    MONGO_DISK_CACHE,
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(data, file)

            except Exception as exc:
                logger.error(
                    "Could not persist MongoDB fallback data (%s).",
                    type(exc).__name__,
                )

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics of MongoDB storage."""
        collections = {
            "matches": self.db.matches.count_documents({}),
            "points": self.db.points.count_documents({}),
            "player_profiles": self.db.player_profiles.count_documents({}),
        }

        engine_name = (
            "MongoDB Atlas (Cloud Cluster)"
            if self.is_live_mongo and self.is_atlas
            else "MongoDB Community Server"
            if self.is_live_mongo
            else "Embedded Document Store (mongomock)"
        )

        return {
            "engine": engine_name,
            "connected": True,
            "is_live": self.is_live_mongo,
            "is_atlas": self.is_atlas,
            "cluster": self.target_cluster,
            "database": self.db_name,
            "connection_status": self.connection_status,
            "connection_error": self.connection_error,
            "collections": collections,
            "total_documents": sum(collections.values()),
            "indexes_active": True,
        }

    def test_connection(self) -> Dict[str, Any]:
        """Verify MongoDB connectivity without exposing raw exceptions."""
        if self.is_live_mongo and self.client:
            try:
                ping_res = self.client.admin.command("ping")
                dbs = self.client.list_database_names()
                cols = self.db.list_collection_names()

                return {
                    "status": "healthy",
                    "engine": (
                        "MongoDB Atlas"
                        if self.is_atlas
                        else "MongoDB Local"
                    ),
                    "cluster": self.target_cluster,
                    "database": self.db_name,
                    "is_live": True,
                    "ping": bool(ping_res.get("ok", 0)),
                    "accessible_databases": dbs,
                    "active_collections": cols,
                    "message": (
                        f"Verified live connection to {self.target_cluster}"
                    ),
                }

            except Exception as exc:
                self.connection_status = "degraded"
                self.connection_error = safe_mongo_error(exc)

                logger.warning(
                    "MongoDB health check failed (%s).",
                    type(exc).__name__,
                )

                return {
                    "status": "degraded",
                    "engine": (
                        "MongoDB Atlas (Ping Failed)"
                        if self.is_atlas
                        else "MongoDB Local (Ping Failed)"
                    ),
                    "cluster": self.target_cluster,
                    "database": self.db_name,
                    "is_live": False,
                    "error": self.connection_error,
                    "message": (
                        "Live connection failed during the health check."
                    ),
                }

        return {
            "status": "fallback_mock",
            "engine": "Embedded Document Store (mongomock)",
            "cluster": self.target_cluster,
            "database": self.db_name,
            "is_live": False,
            "connection_status": self.connection_status,
            "connection_error": self.connection_error,
            "disk_persisted": os.path.exists(MONGO_DISK_CACHE),
            "message": (
                f"Live cluster '{self.target_cluster}' unreachable. "
                "Running with embedded mongomock engine and disk persistence."
            ),
        }


# Singleton instance
mongo_store = MongoStorage()
