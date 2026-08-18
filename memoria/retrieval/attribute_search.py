class AttributeSearch:

    def __init__(self, db):
        self.db = db

    def search(self, subject, attribute):
    if not subject or not attribute:
        return []

    cur = self.conn.execute("""
        SELECT *
        FROM memories
        WHERE subject = ?
          AND attribute = ?
          AND tombstone = 0
    """, (subject, attribute))
    return cur.fetchall()
