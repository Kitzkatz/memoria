from core.logger import debug


class Diagnostics:

    @staticmethod
    def database(db):

        return {
            "rows": db.count(),
            "ok": db.count() >= 0
        }

    @staticmethod
    def vector_store(vs):

        return {
            "vectors": vs.count(),
            "dimension": vs.index.d,
            "ok": vs.count() >= 0
        }

    @staticmethod
    def synchronization(db, vs):

        rows = db.count()
        vectors = vs.count()

        return {
            "db_rows": rows,
            "vectors": vectors,
            "match": rows == vectors
        }

    @staticmethod
    def embedder(embedder):

        vec = embedder.embed("diagnostic")

        return {
            "dimension": len(vec),
            "ok": len(vec) > 0
        }

    @staticmethod
    def llm(llm):

        try:
            if llm is None:
                return {
                    "enabled": False,
                    "ok": True
                }

            return {
                "enabled": True,
                "url": llm.url,
                "ok": True
            }

        except Exception as e:

            return {
                "enabled": False,
                "ok": False,
                "error": str(e)
            }

    @staticmethod
    def full(manager):

        return {
            "database":
                Diagnostics.database(manager.db),

            "vector_store":
                Diagnostics.vector_store(manager.vector_store),

            "sync":
                Diagnostics.synchronization(
                    manager.db,
                    manager.vector_store
                ),

            "embedder":
                Diagnostics.embedder(
                    manager.embedder
                ),

            "llm":
                Diagnostics.llm(
                    manager.llm
                )
        }
