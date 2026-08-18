from memory.models import GoalRecord


class GoalTracker:

    def __init__(self, db):
        self.db = db

    def set_goal(self, goal, progress="started"):

        with self.db.lock:
            cur = self.db.conn.cursor()
            cur.execute(
                """
                INSERT INTO goals (goal, progress, status)
                VALUES (?, ?, ?)
                """,
                (goal, progress, "active")
            )
            self.db.conn.commit()
            return cur.lastrowid

    def update_goal(self, goal_id, progress=None, status=None):

        fields = []
        values = []

        if progress is not None:
            fields.append("progress = ?")
            values.append(progress)

        if status is not None:
            fields.append("status = ?")
            values.append(status)

        if not fields:
            return

        values.append(goal_id)

        with self.db.lock:
            self.db.conn.execute(
                f"UPDATE goals SET {', '.join(fields)} WHERE id = ?",
                values
            )
            self.db.conn.commit()

    def get(self, goal_id):

        cur = self.db.conn.cursor()
        cur.execute(
            "SELECT * FROM goals WHERE id = ?",
            (goal_id,)
        )
        row = cur.fetchone()
        if row:
            return GoalRecord(**dict(row))
        return None

    def list_goals(self, status=None):

        cur = self.db.conn.cursor()

        if status:
            cur.execute(
                "SELECT * FROM goals WHERE status = ?",
                (status,)
            )
        else:
            cur.execute("SELECT * FROM goals")

        return [GoalRecord(**dict(row)) for row in cur.fetchall()]
