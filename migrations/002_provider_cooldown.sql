-- Per-provider circuit-breaker cooldowns for the multi-provider review pool.
--
-- Throttling is enforced on a provider's API key, which is typically shared
-- across repos, so the cooldown is keyed by provider id and is GLOBAL (one row
-- per provider, not per repo). A row is present only while a provider is (or was
-- recently) cooling down; `cooldown_until > now()` means the breaker is open.
CREATE TABLE IF NOT EXISTS provider_cooldown (
    provider       TEXT PRIMARY KEY,
    cooldown_until TIMESTAMPTZ NOT NULL,
    tripped_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason         TEXT
);

CREATE INDEX IF NOT EXISTS provider_cooldown_until_idx
    ON provider_cooldown (cooldown_until);
