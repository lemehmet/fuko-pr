-- Last observed state of each external reviewer bot, per repository.
--
-- Written by the runner at the end of each review (an observation of what
-- CodeRabbit/Copilot did on that PR's HEAD) and read at the start of the next
-- one: a degraded state (rate_limited / paused / unavailable) promotes the
-- config's backup models into the active set for that round (next-round
-- escalation). Keyed per repo -- unlike provider_cooldown, reviewer quotas
-- (CodeRabbit plan limits, Copilot premium-request credits) differ per
-- org/repo, so one repo's throttled reviewer says nothing about another's.
CREATE TABLE IF NOT EXISTS reviewer_health (
    repo        TEXT NOT NULL,
    reviewer    TEXT NOT NULL,
    state       TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    pr          INTEGER,
    detail      TEXT,
    PRIMARY KEY (repo, reviewer)
);
