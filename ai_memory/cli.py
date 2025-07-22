import sys
import json
import click
from .database import MemoryStore, MemoryDatabase
from .memory import Memory
from .model_config import get_model_budget

@click.group()
def cli():
    """AI Memory CLI - manage memories and context."""
    pass

@cli.command()
@click.argument("content")
@click.option("--importance", "-i", default=1.0, type=float, help="Memory importance weight")
@click.option("--conversation-id", "-c", default=None, help="Optional conversation ID")
def add(content, importance, conversation_id):
    """Add a new memory entry."""
    try:
        store = MemoryStore()
        mem = Memory(
            memory_id=None,
            content=content.strip(),
            timestamp=None,
            type="conversation",
            project_id=None,
            importance_weight=importance,
            entities=set(),
            access_count=0,
        )
        if conversation_id:
            mem.project_id = conversation_id
        store.add(mem)
        click.echo("✓ Memory added.")
    except Exception as e:
        click.echo(f"✗ Failed to add memory: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option("--limit", "-n", default=None, type=int, help="Limit the number of results")
@click.option("--conversation-id", "-c", "conv_id", default=None, help="Conversation ID")
@click.option("--contains", "-f", default=None, help="Substring filter for content")
def list(limit, conv_id, contains):

    """List recent memories."""
    try:
        store = MemoryStore()
        cur = store.conn.cursor()
        base = "SELECT mf.conv_id, mf.content, mf.importance, mf.created_at FROM memory_fragments mf"
        joins = []
        query = base
        conditions = []
        params = []
        if conv_id:
            conditions.append("mf.conv_id = ?")
            params.append(conv_id)
        if contains:
            conditions.append("mf.content LIKE ?")
            params.append(f"%{contains}%")
        if entity:
            joins.append("JOIN message_entities me ON mf.msg_id = me.msg_id JOIN entities e ON me.entity_id = e.entity_id")
            conditions.append("e.canonical = ?")
            params.append(entity.lower().strip())
        if joins:
            query += " " + " ".join(joins)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY datetime(mf.created_at) DESC"
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
            content_display = content[:100].strip() + ("…" if len(content) > 100 else "")
            click.echo(f"- [{conv_label}] {content_display} (importance: {importance})")
    except Exception as e:
        click.echo(f"✗ Failed to list memories: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument("query")
@click.option("--model", default="gpt-4", help="Model name to use for context budgeting.")
@click.option("--limit", "-l", "token_limit", type=int, default=None, help="Token limit override for context")
@click.option("--conversation-id", "-c", "conv_id", default=None, help="Optional conversation ID to filter context.")
def context(query, model, token_limit, conv_id):
    """Build and print the optimized context for a given query."""
    try:
        from .memory_optimizer import MemoryOptimizer
        memopt = MemoryOptimizer()
        budget = get_model_budget(model, token_limit)
        context_str = memopt.build_optimal_context(
            {"name": model, "max_tokens": budget}, current_task=query, conversation_id=conv_id
        )
        click.echo(context_str)
    except Exception as e:
        click.echo(f"✗ Failed to build context: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument("output_path", required=False)
@click.option("--conversation-id", "-c", default=None, help="Optional conversation ID")
@click.option("--filter", "-f", "where_clause", default=None, help="Optional SQL WHERE clause")
@click.option("--format", "-t", "format_", type=click.Choice(["json", "text", "table"]), default="json")
def export(output_path, conversation_id, where_clause, format_):
    """Export memories to JSON (optionally filtered by a SQL WHERE clause)."""
    try:
        store = MemoryStore()
        cur = store.conn.cursor()
        base_query = "SELECT * FROM memory_fragments"
        if conversation_id:
            base_query += f" WHERE conv_id = '{conversation_id}'"
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

@cli.command(name="import")
@click.argument("json_path")
def import_(json_path):
    """Import memory data from a structured JSON file."""
    try:
        db = MemoryDatabase("~/ai_memory/ai_memory.db")
        db.import_json(json_path)
        db.close()
        click.echo(f"✓ Imported memory from {json_path}")
    except Exception as e:
        click.echo(f"✗ Failed to import: {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()
