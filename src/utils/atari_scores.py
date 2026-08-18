"""Human / random baseline scores for the Atari 100k benchmark.

Used to turn a raw ``eval/return_mean`` into a **human-normalised score** (HNS)

    hns = (score - random) / (human - random)

so that games on wildly different raw scales (Ms. Pac-Man in the thousands,
Freeway in the tens) become comparable and can be aggregated across games. This
is the primary metric of the Atari 100k literature; logging it per run means the
cross-game aggregate needs no per-game rescaling after the fact.

The constants are the canonical random-agent and human-expert raw scores used
throughout the sample-efficient-RL literature: Wang et al. (2016), "Dueling
Network Architectures for Deep Reinforcement Learning", as tabulated for the
26-game Atari 100k benchmark by Schwarzer et al. (2021), "Data-Efficient
Reinforcement Learning with Self-Predictive Representations" (SPR), Table 3.
Human scores originate with Mnih et al. (2015). Using the community-standard
table (rather than re-measuring) is what makes the numbers comparable to
published Atari 100k results (SimPLe, DrQ, CURL, SPR, ...).

Keys are canonicalised (lower-case, alphanumeric only), so ``human_random`` /
:func:`human_normalized_score` accept any of the spellings that appear in this
codebase — ``"MsPacman"``, ``"mspacman_nec_train"``, ``"ALE/MsPacman-v5"`` —
and return ``None`` for a game not in the benchmark rather than raising.
"""
from __future__ import annotations

import re

#: ``canonical game name -> (random score, human score)`` for the 26 Atari 100k
#: games. Do not edit these numbers casually: they are the published baselines
#: every human-normalised score in the thesis is divided by, so a change here
#: silently rescales every reported result.
_RAW: dict[str, tuple[float, float]] = {
    "Alien":          (227.8,   7127.7),
    "Amidar":         (5.8,     1719.5),
    "Assault":        (222.4,   742.0),
    "Asterix":        (210.0,   8503.3),
    "BankHeist":      (14.2,    753.1),
    "BattleZone":     (2360.0,  37187.5),
    "Boxing":         (0.1,     12.1),
    "Breakout":       (1.7,     30.5),
    "ChopperCommand": (811.0,   7387.8),
    "CrazyClimber":   (10780.5, 35829.4),
    "DemonAttack":    (152.1,   1971.0),
    "Freeway":        (0.0,     29.6),
    "Frostbite":      (65.2,    4334.7),
    "Gopher":         (257.6,   2412.5),
    "Hero":           (1027.0,  30826.4),
    "Jamesbond":      (29.0,    302.8),
    "Kangaroo":       (52.0,    3035.0),
    "Krull":          (1598.0,  2665.5),
    "KungFuMaster":   (258.5,   22736.3),
    "MsPacman":       (307.3,   6951.6),
    "Pong":           (-20.7,   14.6),
    "PrivateEye":     (24.9,    69571.3),
    "Qbert":          (163.9,   13455.0),
    "RoadRunner":     (11.5,    7845.0),
    "Seaquest":       (68.4,    42054.7),
    "UpNDown":        (533.4,   11693.2),
}

#: Tokens that appear in this repo's env-config / run names around the game name
#: (e.g. ``mspacman_nec_train``, ``atari_mfec_eval_rgb``) and must be stripped
#: before matching. Encoder names live here too so ``run.game`` values that a
#: sweep may have folded an encoder into still resolve.
_NON_GAME_TOKENS = frozenset({
    "ale", "v5", "v4", "v0",
    "nec", "mfec", "dqn", "atari",
    "train", "eval", "singleframe",
    "rgb", "gray", "grayscale",
    "rp", "vae", "dinov2", "clip", "mae", "resnet",
})


def _canonical(raw: str) -> str:
    """Reduce any game spelling in this codebase to a lookup key.

    ``"ALE/MsPacman-v5"``, ``"mspacman_nec_train"`` and ``"MsPacman"`` all map
    to ``"mspacman"``. Returns ``""`` when nothing game-like remains.
    """
    s = raw.strip().lower()
    if "/" in s:                      # "ale/mspacman-v5" -> "mspacman-v5"
        s = s.rsplit("/", 1)[1]
    tokens = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    core = [t for t in tokens if t not in _NON_GAME_TOKENS]
    return "".join(core)


#: ``canonical key -> (random, human)`` built once from :data:`_RAW`.
_LOOKUP: dict[str, tuple[float, float]] = {
    _canonical(name): pair for name, pair in _RAW.items()
}


def human_random(game: str) -> tuple[float, float] | None:
    """Return ``(random, human)`` raw scores for ``game``, or ``None`` if it is
    not one of the 26 Atari 100k games. Accepts any spelling used in this repo.
    """
    return _LOOKUP.get(_canonical(game))


def human_normalized_score(game: str, score: float) -> float | None:
    """Human-normalised score for a raw ``score`` on ``game``.

    ``(score - random) / (human - random)``. Returns ``None`` when the game is
    unknown (so the caller can simply skip logging ``eval/hns``) or when the
    human and random baselines coincide (which never happens for the 26 games
    but is guarded so this can never divide by zero).
    """
    ref = human_random(game)
    if ref is None:
        return None
    random_score, human_score = ref
    denom = human_score - random_score
    if denom == 0:
        return None
    return (score - random_score) / denom


def resolve_game(*candidates: str | None) -> str | None:
    """First ``candidate`` that names one of the 26 games, canonicalised.

    Lets a caller pass several config fields in priority order — e.g.
    ``resolve_game(cfg.run.game, cfg.game, cfg.environment.name)`` — and get
    back the canonical key (``"mspacman"``) of the first that matches, or
    ``None`` if none do.
    """
    for candidate in candidates:
        if candidate is None:
            continue
        key = _canonical(str(candidate))
        if key in _LOOKUP:
            return key
    return None
