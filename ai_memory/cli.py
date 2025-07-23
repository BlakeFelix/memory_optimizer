import sys
import json
import click

from .memory_store import MemoryStore
from .database import MemoryDatabase
from .model_config import get_model_budget


@click.group()
def cli():
    """AI Memory CLI - manage memories and context."""
    pass


@cli.command(name="vectorize")
@click.argument("file", type=click.Path(exists=True))
@click.option("--vector-index", required=True, help="Path to FAISS/SQLite index")
@click.option("--model", default="llama3:70b-instruct-q4_K_M")
@click.option("--factory", default="Flat", help="faiss index_factory string")
@click.option(
    "--json-extract",
    type=click.Choice(["auto", "messages", "all", "none"]),
    default="auto",
    help="Extraction mode for JSON files (auto|messages|all|none)",
)
@click.option("--no-meta", is_flag=True, help="Do not write metadata side-car")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def vectorize(file, vector_index, model, factory, json_extract, no_meta, verbose):
    """Embed a file into the vector index."""
    from .vector_embedder import embed_file

    status = embed_file(
        file,
        vector_index,
        model,
        factory=factory,
        json_extract=json_extract,
        no_meta=no_meta,
        verbose=verbose,
    )
    click.echo(f"\u2713 Embedded {file} into {vector_index}")
    if status != 0:
        click.echo("Metadata mismatch", err=True)
        sys.exit(status)

@cli.command()
@click.argument("content")
@click.option("--importance", "-i", default=1.0, type=float, help="Memory importance weight")
@click.option("--conversation-id", "-c", default=None, help="Optional conversation ID")
def add(content, importance, conversation_id):
    """Add a new memory entry."""
    try:
        store = MemoryStore()
        if conversation_id:
            from datetime import datetime, timezone

            ts = datetime.now(tz=timezone.utc).isoformat()
            cur = store.conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO conversations (conv_id, user_id, title, started_at, updated_at) VALUES (?,?,?,?,?)",
                (conversation_id, "default", conversation_id, ts, ts),
            )
        store.add(content.strip(), conv_id=conversation_id, importance=importance)
        click.echo("✓ Memory added.")
    except Exception as e:
        click.echo(f"✗ Failed to add memory: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option("--limit", "-n", default=None, type=int, help="Limit the number of results")
@click.option("--conversation-id", "-c", "conv_id", default=None, help="Conversation ID")
@click.option("--contains", "-f", default=None, help="Substring filter for content")
@click.option("--entity", "-e", default=None, help="Filter by entity value")
def list(limit, conv_id, contains, entity):

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
@click.option("--output", "-o", "output_path", required=False, help="Output file path")
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


@cli.command(name="ingest-zip")
@click.option("--src", default="~/Downloads", help="Directory containing ZIP files")
@click.option("--dest", default="~/chatlogs", help="Extraction destination")
@click.option("--index", default=None, help="Vector index for aimem_bld")
@click.option("--model", default=None, help="Embedding model for aimem_bld")
@click.option("--no-meta", is_flag=True, help="Do not write metadata side-car")
def ingest_zip(src, dest, index, model, no_meta):
    """Scan a directory for chat log ZIPs and import them."""
    try:
        from .ingest import zip_watcher
        args = ["--src", src, "--dest", dest]
        if index:
            args += ["--index", index]
        if model:
            args += ["--model", model]
        if no_meta:
            args += ["--no-meta"]
        zip_watcher.main(args)
    except Exception as e:
        click.echo(f"✗ Failed to ingest ZIPs: {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()
