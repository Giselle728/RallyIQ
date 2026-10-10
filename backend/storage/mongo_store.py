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
import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import mongomock

from backend.config import MONGO_URI, MONGO_DB_NAME, MONGO_DISK_CACHE

logger = logging.getLogger("rallyiq.mongo")

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
        self.client = None
        self.db = None
        self._connect()
        self._setup_indexes()
        self._load_from_disk()
        self._initialized = True

    def _connect(self):
        try:
            # Test live connection with 1.5s timeout
            test_client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=1500)
            test_client.admin.command('ping')
            self.client = test_client
            self.db = self.client[self.db_name]
            self.is_live_mongo = True
            logger.info(f"Connected to live MongoDB at {self.uri} ({self.db_name})")
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            logger.warning(f"Live MongoDB not available ({e}). Using embedded Mongo engine with disk persistence.")
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
        return {
            "engine": "MongoDB Community Server" if self.is_live_mongo else "Embedded Document Store (mongomock)",
            "connected": True,
            "is_live": self.is_live_mongo,
            "database": self.db_name,
            "collections": collections,
            "total_documents": sum(collections.values()),
            "indexes_active": True
        }

# Singleton instance
mongo_store = MongoStorage()
