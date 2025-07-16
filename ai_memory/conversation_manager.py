from ai_memory.conversation_session import ConversationSession

class ConversationManager:
    """Keeps a registry of active sessions."""

    def __init__(self):
        self.sessions = {}

    def get(self, session_id: str | None) -> ConversationSession:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        sess = ConversationSession()
        self.sessions[sess.session_id] = sess
        return sess
