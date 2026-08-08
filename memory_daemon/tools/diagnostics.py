from core.logger import debug, info, error


class Diagnostics:

    @staticmethod
    def database(db):
        """Check database health."""
        try:
            count = db.count()
            return {
                "rows": count,
                "ok": count >= 0,
                "status": "healthy" if count >= 0 else "unknown"
            }
        except Exception as e:
            return {
                "rows": 0,
                "ok": False,
                "status": "error",
                "error": str(e)
            }

    @staticmethod
    def vector_store(vs):
        """Check vector store health."""
        try:
            count = vs.count()
            dim = vs.index.d if hasattr(vs, 'index') else 0
            return {
                "vectors": count,
                "dimension": dim,
                "ok": count >= 0,
                "status": "healthy" if count >= 0 else "unknown"
            }
        except Exception as e:
            return {
                "vectors": 0,
                "dimension": 0,
                "ok": False,
                "status": "error",
                "error": str(e)
            }

    @staticmethod
    def synchronization(db, vs):
        """Check sync between DB and vector store."""
        try:
            rows = db.count()
            vectors = vs.count()
            match = rows == vectors
            return {
                "db_rows": rows,
                "vectors": vectors,
                "match": match,
                "status": "synced" if match else "desynced"
            }
        except Exception as e:
            return {
                "db_rows": 0,
                "vectors": 0,
                "match": False,
                "status": "error",
                "error": str(e)
            }

    @staticmethod
    def embedder(embedder):
        """Check embedder health."""
        try:
            if embedder is None:
                return {
                    "dimension": 0,
                    "ok": False,
                    "status": "not_available"
                }

            vec = embedder.embed("diagnostic")
            return {
                "dimension": len(vec),
                "ok": len(vec) > 0,
                "status": "healthy" if len(vec) > 0 else "unhealthy"
            }
        except Exception as e:
            return {
                "dimension": 0,
                "ok": False,
                "status": "error",
                "error": str(e)
            }

    @staticmethod
    def llm(llm):
        """Check LLM health."""
        try:
            if llm is None:
                return {
                    "enabled": False,
                    "ok": True,
                    "status": "not_configured"
                }

            # Try to get URL if available
            url = getattr(llm, 'url', None)
            if url is None:
                url = getattr(llm, 'base_url', None)
            if url is None:
                url = "unknown"

            return {
                "enabled": True,
                "url": url,
                "ok": True,
                "status": "healthy"
            }

        except Exception as e:
            return {
                "enabled": False,
                "ok": False,
                "status": "error",
                "error": str(e)
            }

    @staticmethod
    def full(manager=None, system=None):
        """
        Run full diagnostics on the system.

        Args:
            manager: Legacy manager object (deprecated)
            system: MemorySystem object (preferred)

        Returns:
            dict: Full diagnostic report
        """
        # Determine which object to use
        if system is not None:
            target = system
        elif manager is not None:
            target = manager
        else:
            return {
                "status": "error",
                "error": "No system or manager provided"
            }

        # Check if target has the expected attributes
        if hasattr(target, 'db') and hasattr(target, 'vector_store'):
            # system object
            db = target.db
            vs = target.vector_store
            embedder = getattr(target, 'embedder', None)
            llm = getattr(target, 'llm', None)
        elif hasattr(target, 'db') and hasattr(target, 'vector_store'):
            # manager object (same structure)
            db = target.db
            vs = target.vector_store
            embedder = getattr(target, 'embedder', None)
            llm = getattr(target, 'llm', None)
        else:
            return {
                "status": "error",
                "error": "Invalid object provided"
            }

        return {
            "database": Diagnostics.database(db),
            "vector_store": Diagnostics.vector_store(vs),
            "sync": Diagnostics.synchronization(db, vs),
            "embedder": Diagnostics.embedder(embedder),
            "llm": Diagnostics.llm(llm),
            "overall_status": "healthy" if all([
                Diagnostics.database(db).get("ok", False),
                Diagnostics.vector_store(vs).get("ok", False),
                Diagnostics.synchronization(db, vs).get("match", False)
            ]) else "degraded"
        }

    @staticmethod
    def quick(manager=None, system=None):
        """Quick diagnostic summary."""
        full_result = Diagnostics.full(manager=manager, system=system)
        return {
            "status": full_result.get("overall_status", "unknown"),
            "db_rows": full_result.get("database", {}).get("rows", 0),
            "vectors": full_result.get("vector_store", {}).get("vectors", 0),
            "synced": full_result.get("sync", {}).get("match", False),
            "llm_enabled": full_result.get("llm", {}).get("enabled", False),
        }
