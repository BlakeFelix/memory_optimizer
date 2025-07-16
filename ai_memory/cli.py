import sys
import json
import click
from datetime import datetime, timezone

from .memory_store import MemoryStore
from .model_config import get_model_budget
from .memory_optimizer import MemoryOptimizer


@click.group()
def cli():
    """AI Memory CLI - manage memories and context."""
    pass


@cli.command()
@click.argument("content")
@click.option("--importance", "-i", type=float, default=1.0, help="Importance score for the memory.")
@click.option("--conversation-id", "-c", "conv_id", default=None, help="Optional conversation ID to associate the memory with.")
def add(content: str, importance: float, conv_id: str):
    """Add a new memory entry."""
    try:
        store = MemoryStore()
        if conv_id:
            cur = store.conn.cursor()
            ts = datetime.now(tz=timezone.utc).isoformat()
            try:
                cur.execute(
                    "INSERT OR IGNORE INTO conversations (conv_id, user_id, title, started_at, updated_at) VALUES (?,?,?,?,?)",
                    (conv_id, "default", "session", ts, ts),
                )
                cur.execute(
                    "UPDATE conversations SET updated_at=? WHERE conv_id=?",
                    (ts, conv_id),
                )
            except Exception as e:
                click.echo(f"✗ Failed to ensure conversation: {e}", err=True)
                store.conn.close()
                sys.exit(1)
        try:
            store.add(content, conv_id=conv_id, importance=importance)
        except Exception as e:
            click.echo(f"✗ Failed to add memory: {e}", err=True)
            store.conn.close()
            sys.exit(1)
        store.conn.close()
        if conv_id:
            click.echo(f"✓ Memory added to conversation {conv_id}.")
        else:
            click.echo("✓ Memory added.")
    except Exception as e:
        click.echo(f"✗ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--limit", "-n", type=int, default=10, help="Number of recent memories to list.")
@click.option("--conversation-id", "-c", "conv_id", default=None, help="Filter by conversation ID.")
@click.option("--contains", "-k", default=None, help="Filter memories that contain a given substring.")
def list(limit: int, conv_id: str, contains: str):
    """List recent memories."""
    try:
        store = MemoryStore()
        cur = store.conn.cursor()
        query = "SELECT conv_id, content, importance, created_at FROM memory_fragments"
        conditions = []
        params = []
        if conv_id:
            conditions.append("conv_id = ?")
            params.append(conv_id)
        if contains:
            conditions.append("content LIKE ?")
            params.append(f"%{contains}%")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY datetime(created_at) DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        store.conn.close()
        if not rows:
            click.echo("No memories found.")
            return
        for row in rows:
            conv_label = row[0] if row[0] else "global"
            content = str(row[1])
            importance = row[2]
            if len(content) > 100:
                content_display = content[:100].strip() + "…"
            else:
                content_display = content
            click.echo(f"- [{conv_label}] {content_display} (importance: {importance})")
    except Exception as e:
        click.echo(f"✗ Failed to list memories: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--model", default="gpt-4", help="Model name to use for context budgeting.")
@click.option("--limit", "-l", "token_limit", type=int, default=None, help="Token limit override for context (if not using model default).")
@click.option("--conversation-id", "-c", "conv_id", default=None, help="Optional conversation ID to filter context.")
def context(query: str, model: str, token_limit: int, conv_id: str):
    """Build and print the optimized context for a given query."""
    try:
        try:
            budget = get_model_budget(model, token_limit)
        except Exception as e:
            click.echo(f"✗ Model config error: {e}", err=True)
            sys.exit(1)
        memopt = MemoryOptimizer()
        context_str = memopt.build_optimal_context({"name": model, "max_tokens": budget}, current_task=query, conversation_id=conv_id)
        if not context_str or context_str.strip() == "":
            click.echo("No relevant memories found for context.")
        else:
            sys.stdout.write(context_str + ("" if context_str.endswith("\n") else "\n"))
    except Exception as e:
        click.echo(f"✗ Failed to build context: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--output", "-o", "output_path", type=str, default=None, help="Output file path for exported JSON.")
@click.option("--where", "where_clause", default=None, help="Optional SQL WHERE clause to filter exported memories.")
def export(output_path: str, where_clause: str):
    """Export memories to JSON (optionally filtered by a SQL WHERE clause)."""
    try:
        store = MemoryStore()
        cur = store.conn.cursor()
        base_query = (
            "SELECT mem_id, conv_id, msg_id, content, importance, token_estimate, created_at, source_type FROM memory_fragments"
        )
        if where_clause:
            clause = where_clause.strip()
            if clause.lower().startswith("where"):
                clause = clause[5:].strip()
            query = f"{base_query} WHERE {clause}"
        else:
            query = base_query
        cur.execute(query)
        cols = [desc[0] for desc in cur.description]
        data = [dict(zip(cols, row)) for row in cur.fetchall()]
        store.conn.close()
        json_data = json.dumps(data, indent=2)
        if output_path:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(json_data)
                click.echo(f"✓ Exported memory to {output_path}")
            except Exception as e:
                click.echo(f"✗ Failed to write file {output_path}: {e}", err=True)
                sys.exit(1)
        else:
            click.echo(json_data)
    except Exception as e:
        click.echo(f"✗ Failed to export memories: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
