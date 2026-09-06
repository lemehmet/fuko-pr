"""Configuration loaded from environment variables (prefix FUKO_) and .env."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the fuko-pr sidecar."""

    model_config = SettingsConfigDict(env_prefix="FUKO_", env_file=".env", extra="ignore")

    database_url: str = ""

    embed_base_url: str = "http://localhost:11434/v1"
    # Also the provenance marker for the stored vectors: db.py re-embeds the
    # whole store when this changes, because two models at the same dimension
    # produce incomparable vectors that nothing else would catch.
    embed_model: str = "qwen3-embedding-0.6b"
    embed_api_key: str | None = None
    embed_dim: int = 1024
    embed_batch_size: int = 32
    # Longest single input sent to the embedder, in characters. Oversized input
    # is not a slow request, it is a failed review: embo 500s anything past its
    # batch rather than truncating it for you.
    #
    # Sized from the batch (8192 tokens) and the *worst* measured density, not
    # the average one. Against embo's tokenizer: Python source and markdown run
    # 3.9 chars/token, but dense JSON hits 1.71 — so an average-case cap would
    # be a cap that only fails on machine-generated text. 12000 chars stays
    # under 8192 tokens for anything denser than 1.47 chars/token, and leaves
    # file digests (6000 chars) and markdown chunks (1500) untouched.
    # Constrained rather than left a bare int because it is used as a slice
    # bound: 0 would embed the empty string and a negative value would cut from
    # the end, so a misconfigured deployment would silently embed nothing
    # instead of failing at startup.
    embed_max_chars: int = Field(default=12000, gt=0)

    # Prefix applied to *queries* only, never to stored documents. Qwen3-
    # Embedding is trained asymmetrically: the query side carries an
    # instruction describing the retrieval task, the document side stays raw,
    # and matching both sides makes retrieval measurably worse rather than
    # better. Set to "" for a symmetric model such as bge-m3.
    embed_query_prefix: str = (
        "Instruct: Given a code review question, retrieve engineering learnings "
        "that apply to it\nQuery: "
    )

    host: str = "0.0.0.0"
    port: int = 8000
    auth_token: str | None = None
    top_k: int = 6
    candidate_k: int = 50
    ingest_max_new: int = 10

    # File digests (#158) ship dark. Population is already opt-in -- nothing
    # writes a digest until someone runs `fuko digest` -- but the knowledge base
    # is shared across repositories and seats, so population alone is the wrong
    # off-switch: one ingest would change what every gating seat reads. Gating
    # retrieval instead means a populated store still reaches nobody until a
    # deployment turns it on, which is what #159's trial seat is for.
    digest_retrieval: bool = False

    # Directory the agentic reviewer writes session transcripts into (#237).
    # Empty means capture is OFF: a default path would have every runner
    # writing NDJSON to a location nobody chose, and the corpus this feeds is
    # kept forever by design. Set it per deployment (a workflow can then upload
    # the file as a run artifact).
    transcript_dir: str = ""

    # Where the SIDECAR puts the transcripts runners ship it (#238). Empty
    # backend means no transcript store, which is the off state and not an
    # error: object storage becomes newly relevant to Postgres deployments with
    # this feature, and one that never configures it has to keep starting and
    # reviewing exactly as before, with transcripts simply absent.
    #
    # Environment rather than `.fuko.toml` because the deployed sidecar has no
    # checkout to read one from -- `docker/Dockerfile.sidecar` copies only
    # `sidecar/` and `migrations/` into `/app`, so `load_config()` finds no file
    # there and `docker/runner-compose.yml` configures the service entirely
    # through FUKO_*. Same argument #216 made for the embedding endpoint.
    #
    # The RUNNER reads these too, but only on the path where no `FUKO_URL` is
    # set (a laptop `fuko review`); a runner pointed at a sidecar ships through
    # it and needs no storage credentials of its own.
    transcript_store_backend: str = ""  # "" (off) | file | s3 | r2
    transcript_store_root: str = ""  # file backend: the directory holding blobs
    transcript_store_bucket: str = ""  # s3/r2
    transcript_store_prefix: str = ""  # s3/r2: key prefix inside the bucket
    transcript_store_endpoint_url: str = ""  # s3/r2: set for R2 and S3-compatibles
    # Names the two credential variables read for s3/r2:
    # <prefix>_ACCESS_KEY_ID and <prefix>_SECRET_ACCESS_KEY (plus <prefix>_REGION).
    # Renaming it needs no source edit: `agentic._store_credential_vars()`
    # derives the two names from THIS setting at run time and feeds them to
    # both the harness-environment strip and the transcript scrub list. That
    # derivation is the mechanism -- `agentic._FUKO_SECRET_VARS`' literal
    # `FUKO_S3_*` entries only cover the default when settings are absent, and
    # deleting the derivation as redundant with them would reopen the leak.
    transcript_store_creds_env_prefix: str = "FUKO_S3"
    # Largest transcript the sidecar will accept in one upload. A ceiling on
    # what a single request can make the process hold, not a policy about
    # transcript size: the epic keeps everything and truncates nothing, so this
    # sits far above any real session (a long agentic review with megabytes of
    # tool results is tens of MB) and exists so a runaway or malformed body
    # cannot exhaust the sidecar's memory. An upload over it is refused whole
    # (413) rather than stored truncated -- a partial blob under a write-once
    # key could never be corrected.
    transcript_max_bytes: int = Field(default=256 * 1024 * 1024, gt=0)


settings = Settings()
