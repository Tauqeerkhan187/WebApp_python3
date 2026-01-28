# Author: TK
# Date: 28/01/2026
# Desc: Mistral embeding, cosine similarity, LLM filter, logging and
# Persistance

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

from .store import load_state, save_state

def now_iso() -> str:

     return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
    """
    Cosine similarity between two vectors.
    Returns value in [-1, 1], usually [0, 1] for embeddings.

    """
    a = np.array(emb1, dtype=np.float32)
    b = np.array(emb2, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

@dataclass
class FriendRecommendation:
    """
    Output object for a recommendation after LLM filtering.
    """
    nickname: str
    message: str
    similarity: float
    reason: str


class FriendForTheMomentService:
    """
    Core app logic:
    - loads/saves state in store.json
    - adds messages (nickname + text + embedding)
    - finds top-3 by cosine similarity
    - asks LLM to filter those top-3
    - logs actions to file
    """

    def __init__(
        self,
        store_path: str = "store.json",
        log_path: str = "app.log",
        embed_model: str = "mistral-embed",
        llm_model: str = "mistral-small-latest",
    ):
        self.store_path = store_path
        self.embed_model = embed_model
        self.llm_model = llm_model

        # ---- Logging setup ----
        self.logger = logging.getLogger("find_a_friend")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not any(
            isinstance(h, logging.FileHandler)
            and h.baseFilename.endswith(log_path)
            for h in self.logger.handlers
        ):
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

        # ---- Load env + init Mistral ----
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("Missing MISTRAL_API_KEY")

        self.client = Mistral(api_key=api_key)

        # ---- Load persistent state ----
        self.state = self._load()

    # =========================
    # Persistence
    # =========================
    def _load(self) -> Dict[str, Any]:
        data = load_state(self.store_path)

        messages = data.get("messages", [])
        counter = int(data.get("counter", 0))

        if not isinstance(messages, list):
            messages = []

        return {
            "messages": messages,
            "counter": counter,
        }

    def _save(self) -> None:
        save_state(self.state, self.store_path)

    def list_messages(self) -> List[Dict[str, Any]]:
        return list(self.state["messages"])

    # =========================
    # Mistral Embeddings
    # =========================
    def embed_text(self, text: str) -> List[float]:
        """
        Compute embedding using mistral-embed
        """
        res = self.client.embeddings.create(
            model=self.embed_model,
            inputs=[text],
        )
        return res.data[0].embedding


    def llm_filter_recommendations(
        self,
        new_nick: str,
        new_text: str,
        top3: List[Tuple[float, Dict[str, Any]]],
    ) -> List[FriendRecommendation]:
        """
        Ask an LLM to decide which of the top-3 are genuinely relevant matches
        (shared intent/thought/activity), and return reasons.

        If LLM fails, we fall back to "all top-3" without reasons.
        """
        if not top3:
            return []

        # We ask the model for strict JSON so it's easy to parse.
        candidates = []
        for sim, item in top3:
            candidates.append(
                {
                    "nickname": item["nickname"],
                    "text": item["text"],
                    "similarity": round(sim, 6),
                }
            )

        system = (
            "You are a helpful assistant that matches people based on short messages. "
            "Only recommend candidates that are truly similar in meaning or intent (same activity/plan/interest/thought). "
            "Return STRICT JSON ONLY."
        )

        user = {
            "new_user": {"nickname": new_nick, "text": new_text},
            "candidates": candidates,
            "task": (
                "From the candidates, pick 0-3 that are genuinely relevant to the new user's message. "
                "If none are relevant, return an empty list. "
                "For each recommended candidate, include a short reason."
            ),
            "output_format": {
                "recommendations": [
                    {"nickname": "string", "reason": "string"}
                ]
            },
        }

        # call Mistral chat API (SDK method names may vary across versions)
        raw = None
        try:
            # Common newer pattern
            resp = self.client.chat.complete(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": str(user)},
                ],
            )
            raw = resp.choices[0].message.content
        except Exception:
            try:
                # Alternate naming
                resp = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": str(user)},
                    ],
                )
                raw = resp.choices[0].message.content
            except Exception as e:
                self.logger.info(f"LLM filter failed, fallback to unfiltered top-3. err={e}")
                raw = None

        # Parse JSON result
        if not raw:
            # fallback: recommend all top-3
            out = []
            for sim, item in top3:
                out.append(
                    FriendRecommendation(
                        nickname=item["nickname"],
                        message=item["text"],
                        similarity=sim,
                        reason="(LLM unavailable) Similar by cosine similarity.",
                    )
                )
            return out

        # parse JSON safely
        import json
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            # attempt to extract JSON chunk
            try:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    parsed = json.loads(raw[start:end + 1])
            except Exception:
                parsed = None

        if not parsed or "recommendations" not in parsed or not isinstance(parsed["recommendations"], list):
            # fallback: recommend all top-3
            out = []
            for sim, item in top3:
                out.append(
                    FriendRecommendation(
                        nickname=item["nickname"],
                        message=item["text"],
                        similarity=sim,
                        reason="(LLM output invalid) Similar by cosine similarity.",
                    )
                )
            return out

        # Map LLM picks back to items
        picks = parsed["recommendations"]
        pick_nicks = {p.get("nickname") for p in picks if isinstance(p, dict)}

        out: List[FriendRecommendation] = []
        for sim, item in top3:
            if item["nickname"] in pick_nicks:
                # get reason
                reason = ""
                for p in picks:
                    if isinstance(p, dict) and p.get("nickname") == item["nickname"]:
                        reason = str(p.get("reason", "")).strip()
                        break
                out.append(
                    FriendRecommendation(
                        nickname=item["nickname"],
                        message=item["text"],
                        similarity=sim,
                        reason=reason or "Similar intent/meaning.",
                    )
                )

        return out

    # Core flow
    def add_message_and_recommend(
        self,
        nickname: str,
        text: str,
    ) -> Tuple[Dict[str, Any], List[Tuple[float, Dict[str, Any]]], List[FriendRecommendation]]:
        """
        Main flow:
        1) embed new text
        2) compute similarity against existing messages
        3) take top-3
        4) ask LLM to filter those top-3
        5) persist new message
        6) log: new message + top-3 similarities
        """
        nickname = nickname.strip()
        text = text.strip()

        if not nickname:
            raise ValueError("Nickname cannot be empty.")
        if not text:
            raise ValueError("Message cannot be empty.")

        #  compute embedding
        emb = self.embed_text(text)

        # compare with existing
        existing = self.state["messages"]
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in existing:
            # item contains "embedding"
            sim = cosine_similarity(emb, item.get("embedding", []))
            scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top3 = scored[:3]

        #  log new message + top-3
        self.logger.info(f"NEW_MESSAGE nickname={nickname!r} text={text!r}")
        if top3:
            for rank, (sim, item) in enumerate(top3, start=1):
                self.logger.info(
                    f"TOP3 rank={rank} sim={sim:.6f} nick={item.get('nickname')!r} text={item.get('text')!r}"
                )
        else:
            self.logger.info("TOP3 none (no existing messages yet)")

        # LLM filtering
        recommendations = self.llm_filter_recommendations(nickname, text, top3)

        # persist new message last (so it doesn't match itself)
        self.state["counter"] += 1
        new_item = {
            "id": self.state["counter"],
            "nickname": nickname,
            "text": text,
            "embedding": emb,           # stored as list (JSON friendly)
            "created_at": now_iso(),
        }
        self.state["messages"].append(new_item)
        self._save()

        return new_item, top3, recommendations





