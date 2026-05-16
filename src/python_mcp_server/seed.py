"""Boot-time DB seeding for vector + graph slices.

Mirrors ems-device-api/src/seed/seed_from_file.ts contract:
  - URL set + marker present  → skip
  - URL set + marker absent   → fetch + restore + write marker
  - URL unset                 → start empty (log + skip)
  - any fetch / restore error → fatal, propagate

Naming follows engine, not deployment — Neo4j cypher dump restores on
Aura cloud and ISO self-hosted alike via the bolt protocol.

Sources (public S3, no auth):
  vector        → vector-cloud.sql.gz | vector-airgapped.sql.gz (pg_dump)
  graph (Neo4j) → graph-neo4j.cypher.gz     (apoc.export.cypher.all output)
  graph (Neptune) → s3://arcnode-public/seed/graph-neptune/{vertices,edges}.csv
                    loaded via Neptune Bulk Loader REST API (sigv4 IAM auth)

Markers:
  vector → arcnode_seed_markers row in postgres, slice='vector'
  graph  → :ArcnodeSeedMarker {slice:'graph'} node (Neo4j or Neptune)
"""

import asyncio
import gzip
import json
import logging
import os
import time
import urllib.request

import asyncpg
import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from neo4j import AsyncGraphDatabase, AsyncManagedTransaction

from .clients.graphiti_client import _split_neo4j_url

logger = logging.getLogger(__name__)

VECTOR_MARKER_SLICE = "vector"
GRAPH_MARKER_SLICE = "graph"

# Neptune Bulk Loader payload constants — see
# https://docs.aws.amazon.com/neptune/latest/userguide/bulk-load.html
# graph-neptune-v2/ has edges.csv stripped of multi-valued []-typed columns
# (Neptune doesn't allow multi-valued props on edges, even though Graphiti's
# default CSV export emits them). Vertices.csv is unchanged from the
# original — vertex multi-vals are fine. Original kept at graph-neptune/
# for Aura/self-hosted Neo4j paths that don't have this constraint.
NEPTUNE_S3_SOURCE = "s3://arcnode-public/seed/graph-neptune-v2/"
NEPTUNE_LOAD_POLL_INTERVAL_SEC = 10
NEPTUNE_LOAD_MAX_WAIT_SEC = 1800  # 30 min — bulk load of ~400MB is < 5 min typical
# Terminal load states; everything else is in-progress.
_NEPTUNE_LOAD_DONE_STATES = {
    "LOAD_COMPLETED",
    "LOAD_FAILED",
    "LOAD_CANCELLED_BY_USER",
    "LOAD_CANCELLED_DUE_TO_ERRORS",
}


def _fetch_gunzip(url: str) -> str:
    """Fetch + gunzip a public S3 .gz artifact, return decoded text."""
    with urllib.request.urlopen(url) as resp:  # nosec B310  # noqa: S310
        return gzip.decompress(resp.read()).decode()


async def _psql_apply(dump_url: str, conn_url: str) -> None:
    """Stream the gzipped dump through `psql -f -` for restore.

    asyncpg.execute can't reliably run a multi-statement pg_dump:
    psql metacommands (\\restrict), COPY ... FROM stdin, dollar-quoted
    function bodies, and embedded newlines in INSERT VALUES all need
    a libpq client (psql), not asyncpg's simple query path. psql
    handles all of it natively.

    Requires `psql` binary in PATH — see consumer Dockerfile (must
    install postgresql-client alongside python).
    """
    with urllib.request.urlopen(dump_url) as resp:  # nosec B310  # noqa: S310
        sql = gzip.decompress(resp.read())
    proc = await asyncio.create_subprocess_exec(
        "psql",
        "-v",
        "ON_ERROR_STOP=1",
        "--single-transaction",
        "-d",
        conn_url,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate(sql)
    if proc.returncode != 0:
        raise RuntimeError(
            f"psql restore failed (rc={proc.returncode}): "
            f"{stderr.decode(errors='replace')[-500:]}"
        )


async def seed_vector(seed_url: str) -> None:
    """Restore vector dump into VECTOR_URL postgres if marker absent.

    Restore via psql -f (subprocess) — see _psql_apply for why asyncpg
    can't run pg_dump output directly. Marker write goes through
    asyncpg afterwards so we can use a parameterized INSERT.

    Failure modes:
      - psql exits non-zero → RuntimeError; marker not written; safe to
        retry (psql will hit "already exists" on re-create unless the
        caller cleans up; recommended retry path = manually
        `DROP SCHEMA public CASCADE; CREATE SCHEMA public;`)
    """
    conn_url = os.environ["VECTOR_URL"]
    conn = await asyncpg.connect(conn_url)
    try:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS arcnode_seed_markers ("
            "slice TEXT PRIMARY KEY, seeded_at TIMESTAMPTZ DEFAULT now())"
        )
        marker = await conn.fetchval(
            "SELECT 1 FROM arcnode_seed_markers WHERE slice=$1",
            VECTOR_MARKER_SLICE,
        )
        if marker:
            logger.info("vector slice already seeded; skipping")
            return
        logger.info("seeding vector slice from %s", seed_url)
        await _psql_apply(seed_url, conn_url)
        await conn.execute(
            "INSERT INTO arcnode_seed_markers (slice) VALUES ($1) "
            "ON CONFLICT DO NOTHING",
            VECTOR_MARKER_SLICE,
        )
        logger.info("vector slice seeded")
    finally:
        await conn.close()


async def seed_graph_neo4j(seed_url: str) -> None:
    """Restore cypher dump into GRAPH_URL Neo4j if marker absent.

    Restore + marker write run in one managed transaction via
    session.execute_write — partial failure rolls back so a retry
    starts from a clean state.
    """
    uri, user, password = _split_neo4j_url(os.environ["GRAPH_URL"])
    if user is None or password is None:
        raise RuntimeError("GRAPH_URL must include user:password (Aura format)")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        async with driver.session() as session:
            result = await session.run(
                "MATCH (m:ArcnodeSeedMarker {slice: $slice}) RETURN m LIMIT 1",
                slice=GRAPH_MARKER_SLICE,
            )
            if await result.single():
                logger.info("graph slice already seeded; skipping")
                return
        logger.info("seeding graph slice from %s", seed_url)
        cypher_script = _fetch_gunzip(seed_url)
        statements = [s.strip() for s in cypher_script.split(";\n") if s.strip()]

        async def _restore(tx: AsyncManagedTransaction) -> None:
            for stmt in statements:
                # Reason: stmt is from our own dump file — driver's
                # LiteralString constraint is for user-input safety,
                # doesn't apply to a controlled artifact.
                await tx.run(stmt)  # ty: ignore[invalid-argument-type]
            await tx.run(
                "MERGE (m:ArcnodeSeedMarker {slice: $slice}) "
                "ON CREATE SET m.seeded_at = datetime()",
                slice=GRAPH_MARKER_SLICE,
            )

        async with driver.session() as session:
            await session.execute_write(_restore)
        logger.info("graph slice seeded")
    finally:
        await driver.close()


def _sigv4_headers(method: str, url: str, body: bytes = b"") -> dict[str, str]:
    """Sign a request with SigV4 against the neptune-db service.

    Returns the headers to attach to the actual httpx call. Uses the
    default boto3 credential chain → EC2 instance role in prod.
    """
    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        raise RuntimeError("no AWS credentials available for Neptune sigv4")
    region = os.environ.get("AWS_REGION", session.region_name or "us-east-1")
    req = AWSRequest(method=method, url=url, data=body)
    SigV4Auth(creds, "neptune-db", region).add_auth(req)
    return dict(req.headers)


async def _neptune_opencypher(host: str, query: str) -> dict:
    """POST an openCypher query to a Neptune cluster; return JSON results."""
    url = f"https://{host}:8182/opencypher"
    body = json.dumps({"query": query}).encode()
    headers = _sigv4_headers("POST", url, body)
    headers["Content-Type"] = "application/json"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, content=body, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _neptune_has_marker(host: str) -> bool:
    """Return True if the ArcnodeSeedMarker {slice:'graph'} node exists."""
    result = await _neptune_opencypher(
        host,
        "MATCH (m:ArcnodeSeedMarker) WHERE m.slice = 'graph' RETURN m LIMIT 1",
    )
    return bool(result.get("results"))


async def _neptune_write_marker(host: str) -> None:
    """Set the seed marker via openCypher MERGE."""
    await _neptune_opencypher(
        host,
        "MERGE (m:ArcnodeSeedMarker {slice: 'graph'}) "
        "ON CREATE SET m.seeded_at = datetime()",
    )


async def _neptune_start_load(host: str, loader_role_arn: str) -> str:
    """Kick off the Bulk Loader; return loadId."""
    url = f"https://{host}:8182/loader"
    region = os.environ.get("AWS_REGION", "us-east-1")
    body = json.dumps(
        {
            "source": NEPTUNE_S3_SOURCE,
            "format": "csv",
            "iamRoleArn": loader_role_arn,
            "region": region,
            # Tolerate per-row errors — the loader logs them via the
            # `errors` endpoint. The 1.5GB CSVs have a long-tail of
            # malformed rows that won't be fixed at the source; bailing
            # the whole load on a single parse error makes the seed
            # impossible. (Phase 8 v4 hit exactly this: 1 error in
            # 3.1M rows cancelled the entire load.)
            "failOnError": "FALSE",
            "parallelism": "MEDIUM",
            "updateSingleCardinalityProperties": "FALSE",
            "queueRequest": "TRUE",
        }
    ).encode()
    headers = _sigv4_headers("POST", url, body)
    headers["Content-Type"] = "application/json"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, content=body, headers=headers)
        resp.raise_for_status()
        payload = resp.json()["payload"]
        return payload["loadId"]


async def _neptune_wait_for_load(host: str, load_id: str) -> None:
    """Poll loader status until terminal; raise on failure."""
    url = f"https://{host}:8182/loader/{load_id}"
    deadline = time.monotonic() + NEPTUNE_LOAD_MAX_WAIT_SEC
    while True:
        headers = _sigv4_headers("GET", url)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            overall = resp.json()["payload"]["overallStatus"]
            status = overall["status"]
        if status == "LOAD_COMPLETED":
            logger.info("neptune load %s complete", load_id)
            return
        if status in _NEPTUNE_LOAD_DONE_STATES:
            raise RuntimeError(f"neptune load {load_id} ended with {status}: {overall}")
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"neptune load {load_id} did not complete within "
                f"{NEPTUNE_LOAD_MAX_WAIT_SEC}s; last status: {status}"
            )
        logger.info("neptune load %s status=%s; polling", load_id, status)
        await asyncio.sleep(NEPTUNE_LOAD_POLL_INTERVAL_SEC)


async def seed_graph_neptune() -> None:
    """Load pre-baked CSVs into Neptune via Bulk Loader if marker absent.

    Reads NEPTUNE_HOST + NEPTUNE_LOADER_ROLE_ARN from env (CFN UserData
    writes these from SSM Parameter Store). The S3 source prefix is
    hardcoded — pre-baked artifacts at arcnode-public/seed/graph-neptune/.

    Idempotent via :ArcnodeSeedMarker {slice:'graph'} node in Neptune
    itself — same pattern as Neo4j. Re-running this fn on a seeded
    cluster is a no-op (one openCypher check + return).
    """
    host = os.environ["NEPTUNE_HOST"]
    loader_role_arn = os.environ["NEPTUNE_LOADER_ROLE_ARN"]

    if await _neptune_has_marker(host):
        logger.info("graph (neptune) slice already seeded; skipping")
        return
    logger.info(
        "seeding graph (neptune) slice from %s via role %s",
        NEPTUNE_S3_SOURCE,
        loader_role_arn,
    )
    load_id = await _neptune_start_load(host, loader_role_arn)
    await _neptune_wait_for_load(host, load_id)
    await _neptune_write_marker(host)
    logger.info("graph (neptune) slice seeded")


async def seed_all(vector_url: str | None, graph_neo4j_url: str | None) -> None:
    """Run available seeds at boot. Skips a slice when its env conn is unset.

    Graph backend is chosen by env shape:
      - GRAPH_URL  (Aura / self-hosted Neo4j) → seed_graph_neo4j
      - NEPTUNE_HOST + NEPTUNE_LOADER_ROLE_ARN (defense) → seed_graph_neptune

    Defense customers should have NEPTUNE_HOST but no GRAPH_URL.
    Commercial customers have GRAPH_URL but no NEPTUNE_HOST. Airgapped
    customers have GRAPH_URL pointing at their self-hosted Neo4j.
    """
    if vector_url and os.environ.get("VECTOR_URL"):
        await seed_vector(vector_url)
    else:
        logger.info("vector seed skipped (no VECTOR_URL or seed URL)")
    if graph_neo4j_url and os.environ.get("GRAPH_URL"):
        await seed_graph_neo4j(graph_neo4j_url)
    elif os.environ.get("NEPTUNE_HOST") and os.environ.get("NEPTUNE_LOADER_ROLE_ARN"):
        await seed_graph_neptune()
    else:
        logger.info(
            "graph seed skipped (no GRAPH_URL+seed-url, "
            "no NEPTUNE_HOST+NEPTUNE_LOADER_ROLE_ARN)"
        )
