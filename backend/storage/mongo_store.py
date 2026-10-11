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
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure
import mongomock

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
                creds, host = netloc.split("@", 1)
                masked_netloc = "****:****@" + host
                return uri.replace(netloc, masked_netloc)
        return uri
    except Exception:
        return "mongodb+srv://****:****@<masked-cluster>"

class MongoStorage:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MongoStorage, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB_NAME):
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
        try:
            connect_kwargs: Dict[str, Any] = {
                "serverSelectionTimeoutMS": 4000,
                "connectTimeoutMS": 4000,
                "socketTimeoutMS": 4000
            }
            # Add certifi CA bundle for Atlas TLS verification (never disable TLS)
            if self.is_atlas or "ssl=true" in self.uri.lower() or "tls=true" in self.uri.lower():
                connect_kwargs["tlsCAFile"] = certifi.where()

            test_client = pymongo.MongoClient(self.uri, **connect_kwargs)
            # Authenticate and ping cluster
            test_client.admin.command('ping')

            self.client = test_client
            self.db = self.client[self.db_name]
            self.is_live_mongo = True
            self.connection_status = "connected_atlas" if self.is_atlas else "connected_local"
            self.connection_error = None
            logger.info(f"Connected to live MongoDB cluster [{self.target_cluster}] (database: {self.db_name})")
        except Exception as e:
            raw_err = str(e)
            # Mask any credentials that might appear in error strings
            safe_err = raw_err
            if "@" in safe_err and ":" in safe_err:
                try:
                    p = urlparse(self.uri)
                    if p.password and p.password in safe_err:
                        safe_err = safe_err.replace(p.password, "********")
                except Exception:
                    pass
            self.connection_error = safe_err
            self.connection_status = "fallback_mock"
            logger.warning(
                f"Live MongoDB cluster [{self.target_cluster}] not available ({safe_err[:180]}). "
                "Engaging embedded Document Store (mongomock with disk sync)."
            )
            self.client = mongomock.MongoClient()
            self.db = self.client[self.db_name]
            self.is_live_mongo = False

    def _setup_indexes(self):
        """Create useful indexes on collections."""
        try:
            self.db.matches.create_index([("match_id", pymongo.ASCENDING)], unique=True)
            self.db.matches.create_index([("Player 1", pymongo.ASCENDING)])
            self.db.matches.create_index([("Player 2", pymongo.ASCENDING)])
            self.db.matches.create_index([("Surface", pymongo.ASCENDING)])
            self.db.matches.create_index([("Year", pymongo.ASCENDING)])

            self.db.points.create_index([("match_id", pymongo.ASCENDING), ("Pt", pymongo.ASCENDING)])

            self.db.player_profiles.create_index([("player", pymongo.ASCENDING)], unique=True)

            self.db.stream_events.create_index([("event_id", pymongo.ASCENDING)], unique=True)
            self.db.stream_events.create_index([("match_id", pymongo.ASCENDING)])
            self.db.stream_events.create_index([("timestamp", pymongo.ASCENDING)])

            self.db.stream_metrics.create_index([("timestamp", pymongo.ASCENDING)])
            logger.info("MongoDB indexes successfully created.")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")

    def _load_from_disk(self):
        """Load cached documents from disk if using embedded mongomock."""
        if not self.is_live_mongo and os.path.exists(MONGO_DISK_CACHE):
            try:
                with open(MONGO_DISK_CACHE, "r") as f:
                    data = json.load(f)
                    for col_name, docs in data.items():
                        if docs:
                            self.db[col_name].drop()
                            self.db[col_name].insert_many(docs)
                logger.info(f"Loaded existing documents from {MONGO_DISK_CACHE}")
            except Exception as e:
                logger.warning(f"Could not load disk cache: {e}")

    def persist_to_disk(self):
        """Persist current collections to disk."""
        if not self.is_live_mongo:
            try:
                data = {}
                for col_name in ["matches", "player_profiles", "stream_events", "stream_metrics"]:
                    cursor = self.db[col_name].find()
                    docs = []
                    for doc in cursor:
                        doc_copy = dict(doc)
                        if "_id" in doc_copy:
                            doc_copy["_id"] = str(doc_copy["_id"])
                        docs.append(doc_copy)
                    data[col_name] = docs
                with open(MONGO_DISK_CACHE, "w") as f:
                    json.dump(data, f)
            except Exception as e:
                logger.error(f"Error persisting Mongo to disk: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics of MongoDB storage."""
        collections = {
            "matches": self.db.matches.count_documents({}),
            "points": self.db.points.count_documents({}),
            "player_profiles": self.db.player_profiles.count_documents({}),
            "stream_events": self.db.stream_events.count_documents({}),
            "stream_metrics": self.db.stream_metrics.count_documents({})
        }
        engine_name = (
            "MongoDB Atlas (Cloud Cluster)" if self.is_live_mongo and self.is_atlas
            else "MongoDB Community Server" if self.is_live_mongo
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
            "indexes_active": True
        }

    def test_connection(self) -> Dict[str, Any]:
        """Safely verify active MongoDB connectivity without exposing credentials."""
        if self.is_live_mongo and self.client:
            try:
                ping_res = self.client.admin.command('ping')
                dbs = self.client.list_database_names()
                cols = self.db.list_collection_names()
                return {
                    "status": "healthy",
                    "engine": "MongoDB Atlas" if self.is_atlas else "MongoDB Local",
                    "cluster": self.target_cluster,
                    "database": self.db_name,
                    "is_live": True,
                    "ping": bool(ping_res.get("ok", 0)),
                    "accessible_databases": dbs,
                    "active_collections": cols,
                    "message": f"Verified live connection to {self.target_cluster}"
                }
            except Exception as e:
                return {
                    "status": "degraded",
                    "engine": "MongoDB Atlas (Ping Failed)",
                    "cluster": self.target_cluster,
                    "database": self.db_name,
                    "is_live": False,
                    "error": str(e)[:200],
                    "message": "Live connection was initialized but command ping failed."
                }
        else:
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
                )
            }

# Singleton instance
mongo_store = MongoStorage()
