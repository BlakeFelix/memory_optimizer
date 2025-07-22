from flask import Flask, request, jsonify, Response
from .conversation_manager import ConversationManager
from .conversation_session import ConversationSession
from .memory_store import MemoryStore
from .model_config import get_model_budget

app = Flask(__name__)
conv_manager = ConversationManager()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route('/memory', methods=['POST'])
def add_memory():
    """Add a new memory via JSON input."""
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400
    if not data or "content" not in data:
        return jsonify({"error": "Content is required."}), 400
    content = data.get("content", "")
    importance = data.get("importance", 1.0)
    conv_id = data.get("conversation_id", None)
    try:
        importance = float(importance)
    except Exception:
        return jsonify({"error": "Importance must be a number."}), 400
    try:
        if conv_id:
            session = conv_manager.sessions.get(conv_id)
            if not session:
                session = ConversationSession()
                session.session_id = conv_id
                conv_manager.sessions[conv_id] = session
            cur = session.optimizer.store.conn.cursor()
            ts = session.start_ts
            from datetime import datetime

            ts_iso = datetime.fromtimestamp(ts).astimezone().isoformat()
            cur.execute(
                "INSERT OR IGNORE INTO conversations (conv_id, user_id, title, started_at, updated_at) VALUES (?,?,?,?,?)",
                (conv_id, "default", session.title, ts_iso, ts_iso),
            )
            cur.execute(
                "UPDATE conversations SET updated_at=? WHERE conv_id=?",
                (datetime.now().astimezone().isoformat(), conv_id),
            )
            mem_id = session.optimizer.store.add(
                content, conv_id=conv_id, importance=importance
            )
        else:
            store = MemoryStore()
            mem_id = store.add(content, conv_id=None, importance=importance)
            store.conn.close()
        return jsonify({"status": "success", "mem_id": mem_id}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to add memory: {e}"}), 500


@app.route('/context', methods=['POST'])
def build_context():
    """Build optimized context via JSON input."""
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400
    if not data or "query" not in data:
        return jsonify({"error": "Query is required."}), 400
    query = data.get("query", "")
    model = data.get("model", "gpt-4")
    token_limit = data.get("token_limit", None)
    conv_id = data.get("conversation_id", None)
    try:
        budget = get_model_budget(model, token_limit)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    try:
        if conv_id:
            session = conv_manager.sessions.get(conv_id)
            if not session:
                session = ConversationSession()
                session.session_id = conv_id
                conv_manager.sessions[conv_id] = session
            context_str = session.build_context(query, model=model, limit=budget)
        else:
            from .memory_optimizer import MemoryOptimizer

            memopt = MemoryOptimizer()
            context_str = memopt.build_optimal_context(
                {"name": model, "max_tokens": budget}, current_task=query
            )
        return jsonify({"context": context_str}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to build context: {e}"}), 500


@app.route('/export/<fmt>', methods=['GET'])
def export_memories(fmt):
    """Export memories in specified format (json or markdown)."""
    fmt = fmt.lower()
    if fmt not in ("json", "markdown"):
        return jsonify({"error": "Format not supported."}), 400
    try:
        store = MemoryStore()
        data = list(store.get_all().values())
        store.conn.close()
        if fmt == "json":
            serialized = []
            for m in data:
                d = vars(m).copy()
                if isinstance(d.get("entities"), set):
                    d["entities"] = list(d["entities"])
                serialized.append(d)
            return jsonify(serialized), 200
        elif fmt == "markdown":
            lines = []
            for mem in data:
                content_line = mem.content.replace("\n", " ")
                lines.append(f"- {content_line}")
            md_text = "\n".join(lines)
            return Response(md_text, mimetype="text/markdown")
    except Exception as e:
        return jsonify({"error": f"Failed to export memories: {e}"}), 500


if __name__ == "__main__":
    app.run(host="localhost", port=5678)
