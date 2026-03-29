from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "before",
    "after",
    "near",
    "another",
    "other",
    "potential",
    "look",
}

_CLOTHING_TERMS = (
    "shirt",
    "jacket",
    "coat",
    "hoodie",
    "vest",
    "uniform",
    "pants",
    "dress",
    "hat",
    "helmet",
)
_SCENE_TERMS = {
    "intersection",
    "shop",
    "store",
    "sidewalk",
    "road",
    "street",
    "restaurant",
    "terrace",
    "parking",
    "driveway",
    "workbench",
    "site",
}
_RELATION_TO_QUERY = (
    (("attack", "attacked", "assault", "hit", "kick", "punch", "lunge", "fight", "altercation"), "physical struggle"),
    (("snatch", "rob", "robbery", "steal", "grab bag"), "bag snatch"),
    (("fall", "falls", "falling", "fell", "trip", "stumble"), "person falling"),
    (("ground", "lying", "fallen"), "person on ground"),
    (("fire", "flame", "flames"), "fire"),
    (("smoke",), "smoke"),
)


def normalize_query_text(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\b(?:the|a|an)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in normalize_query_text(text).split()
        if token and token not in _STOPWORDS
    }


def _classify_query_kind(text: str) -> str:
    tokens = _tokenize(text)
    if not tokens:
        return "object"
    if tokens & _SCENE_TERMS:
        return "scene_context"
    if any(term in text for term in _CLOTHING_TERMS):
        return "attribute_object"
    if any(keyword in text for keywords, _ in _RELATION_TO_QUERY for keyword in keywords):
        return "event_relation"
    return "object"


def _append_query_entry(entries: list[dict[str, Any]], text: str, *, kind: Optional[str] = None, weight: float = 1.0) -> None:
    normalized = normalize_query_text(text)
    if not normalized:
        return
    resolved_kind = kind or _classify_query_kind(normalized)
    for entry in entries:
        if entry["text"] == normalized:
            entry["weight"] = max(float(entry.get("weight") or 0.0), float(weight))
            if entry.get("kind") == "object" and resolved_kind != "object":
                entry["kind"] = resolved_kind
            return
    entries.append(
        {
            "text": normalized,
            "kind": resolved_kind,
            "weight": float(weight),
        }
    )


def normalize_key_object_phrases(key_objects: Sequence[str]) -> List[Dict[str, Any]]:
    normalized_queries: List[Dict[str, Any]] = []
    for raw_text in key_objects or []:
        normalized = normalize_query_text(raw_text)
        if not normalized:
            continue

        # Split compact conjunctions like "pedestrians and vehicles".
        for piece in re.split(r"\s+(?:and|/)\s+", normalized):
            if piece and len(piece.split()) <= 4:
                _append_query_entry(normalized_queries, piece, weight=0.9)

        clothing_pattern = re.compile(
            rf"\b(person|man|woman|individual|victim|attacker)\s+in\s+([a-z0-9 ]{{1,40}}?(?:{'|'.join(_CLOTHING_TERMS)}))\b"
        )
        for match in clothing_pattern.finditer(normalized):
            subject = match.group(1)
            if subject in {"victim", "attacker", "individual"}:
                subject = "person"
            _append_query_entry(normalized_queries, f"{subject} in {match.group(2)}", kind="attribute_object", weight=1.0)

        if len(normalized.split()) <= 4:
            _append_query_entry(normalized_queries, normalized, weight=1.0)

        for keywords, query_text in _RELATION_TO_QUERY:
            if any(keyword in normalized for keyword in keywords):
                weight = 0.8 if query_text not in {"physical struggle", "person on ground"} else 0.9
                _append_query_entry(normalized_queries, query_text, kind="event_relation", weight=weight)

    return normalized_queries


def build_proposal_supervision(
    *,
    key_objects: Sequence[str],
    evidence_moments: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    queries: List[Dict[str, Any]] = []
    moments = [dict(moment) for moment in (evidence_moments or [])]
    for index, raw_text in enumerate(key_objects or [], start=1):
        normalized_queries = normalize_key_object_phrases([str(raw_text or "")])
        linked_moments: List[Dict[str, Any]] = []
        normalized_tokens = [_tokenize(entry["text"]) for entry in normalized_queries]
        for moment in moments:
            role = str(moment.get("role") or "")
            moment_tokens = _tokenize(f"{moment.get('description') or ''} {role}")
            best_overlap = 0.0
            for query_tokens in normalized_tokens:
                if not query_tokens:
                    continue
                overlap = len(query_tokens & moment_tokens) / float(max(len(query_tokens), 1))
                best_overlap = max(best_overlap, overlap)
            if best_overlap <= 0.0:
                continue
            linked_moments.append(moment)
        queries.append(
            {
                "query_id": f"pq{index}",
                "raw_text": str(raw_text or ""),
                "normalized_queries": normalized_queries,
                "linked_moment_ids": [str(moment.get("moment_id") or "") for moment in linked_moments if moment.get("moment_id")],
                "linked_roles": [str(moment.get("role") or "") for moment in linked_moments if moment.get("role")],
                "linked_windows_sec": [
                    [float(moment.get("start_sec") or 0.0), float(moment.get("end_sec") or 0.0)]
                    for moment in linked_moments
                ],
                "alignment_source": "weak_alignment",
            }
        )
    return {
        "queries": queries,
        "has_oracle_supervision": bool(queries),
    }


def select_query_for_moment(
    proposal_supervision: Optional[Dict[str, Any]],
    *,
    moment: Dict[str, Any],
    fallback_query: str,
) -> tuple[str, str]:
    if not isinstance(proposal_supervision, dict):
        return fallback_query, "role_fallback"
    moment_id = str(moment.get("moment_id") or "")
    role = str(moment.get("role") or "")
    moment_tokens = _tokenize(f"{moment.get('description') or ''} {role}")
    best_text = normalize_query_text(fallback_query) or str(fallback_query or "")
    best_source = "role_fallback"
    best_score = -1.0
    for query_group in proposal_supervision.get("queries") or []:
        linked_moment_ids = {str(value) for value in query_group.get("linked_moment_ids") or [] if value}
        linked_roles = {str(value) for value in query_group.get("linked_roles") or [] if value}
        if moment_id and linked_moment_ids and moment_id not in linked_moment_ids and role not in linked_roles:
            continue
        for entry in query_group.get("normalized_queries") or []:
            text = str(entry.get("text") or "")
            if not text:
                continue
            query_tokens = _tokenize(text)
            overlap = len(query_tokens & moment_tokens) / float(max(len(query_tokens), 1)) if query_tokens else 0.0
            score = float(entry.get("weight") or 0.0) + overlap
            if score > best_score:
                best_score = score
                best_text = text
                best_source = "oracle_key_objects"
    return best_text, best_source


def compute_frame_cache_signature(*, fps: float, frame_indices: Sequence[int], num_frames: int) -> str:
    payload = {
        "fps": round(float(fps), 6),
        "frame_indices": [int(index) for index in frame_indices],
        "num_frames": int(num_frames),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def coerce_feature_cache_payload(
    payload: Any,
    *,
    fps: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    if isinstance(payload, torch.Tensor):
        resolved_frame_indices = [int(index) for index in (frame_indices or range(int(payload.shape[0])))]
        resolved_fps = float(fps or 1.0)
        timestamps = [round(float(index) / max(resolved_fps, 1e-6), 6) for index in range(len(resolved_frame_indices))]
        return {
            "version": "saver_feature_cache_v0",
            "model_name": "legacy_tensor",
            "fps": resolved_fps,
            "frame_indices": resolved_frame_indices,
            "timestamps_sec": timestamps,
            "embeddings": payload,
            "embedding_dim": int(payload.shape[-1]) if payload.ndim >= 2 else 1,
            "normalized": False,
            "frame_cache_signature": compute_frame_cache_signature(
                fps=resolved_fps,
                frame_indices=resolved_frame_indices,
                num_frames=len(resolved_frame_indices),
            ),
        }
    if not isinstance(payload, dict):
        return None
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, torch.Tensor):
        resolved_frame_indices = [int(index) for index in (payload.get("frame_indices") or frame_indices or range(int(embeddings.shape[0])))]
        resolved_fps = float(payload.get("fps") or fps or 1.0)
        timestamps = payload.get("timestamps_sec")
        if not isinstance(timestamps, list) or len(timestamps) != len(resolved_frame_indices):
            timestamps = [round(float(index) / max(resolved_fps, 1e-6), 6) for index in range(len(resolved_frame_indices))]
        return {
            "version": str(payload.get("version") or "saver_feature_cache_v1"),
            "model_name": str(payload.get("model_name") or "unknown"),
            "fps": resolved_fps,
            "frame_indices": resolved_frame_indices,
            "timestamps_sec": [float(value) for value in timestamps],
            "embeddings": embeddings,
            "embedding_dim": int(payload.get("embedding_dim") or (embeddings.shape[-1] if embeddings.ndim >= 2 else 1)),
            "normalized": bool(payload.get("normalized", False)),
            "frame_cache_signature": str(
                payload.get("frame_cache_signature")
                or compute_frame_cache_signature(
                    fps=resolved_fps,
                    frame_indices=resolved_frame_indices,
                    num_frames=len(resolved_frame_indices),
                )
            ),
        }
    return None


def _l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    denom = tensor.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return tensor / denom


def coerce_encoder_feature_tensor(
    value: Any,
    *,
    preferred_keys: Sequence[str],
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value

    def _extract(candidate: Any) -> Optional[torch.Tensor]:
        if isinstance(candidate, torch.Tensor):
            return candidate
        if isinstance(candidate, (list, tuple)) and candidate:
            for item in reversed(candidate):
                if isinstance(item, torch.Tensor):
                    return item
        return None

    for key in preferred_keys:
        candidate = None
        if isinstance(value, dict) and key in value:
            candidate = value.get(key)
        elif hasattr(value, key):
            candidate = getattr(value, key)
        tensor = _extract(candidate)
        if tensor is None:
            continue
        if key in {"last_hidden_state", "hidden_state", "hidden_states"} and tensor.ndim >= 3:
            return tensor.mean(dim=1)
        return tensor

    raise TypeError(
        "Could not coerce encoder output to a feature tensor. "
        f"Tried keys={list(preferred_keys)} on type={type(value).__name__}."
    )


def _cluster_sorted_indices(indices: Sequence[int], *, max_gap: int) -> List[List[int]]:
    clusters: List[List[int]] = []
    for index in sorted(int(value) for value in indices):
        if not clusters or index - clusters[-1][-1] > max_gap:
            clusters.append([index])
        else:
            clusters[-1].append(index)
    return clusters


def feature_guided_frame_proposal(
    *,
    feature_cache: Optional[Dict[str, Any]],
    proposal_runtime: Any,
    query: str,
    start_sec: float,
    end_sec: float,
    fps: float,
    num_frames: int,
    top_k_candidates: int = 8,
    candidate_merge_gap_sec: float = 1.0,
    query_source: str = "model",
) -> Dict[str, Any]:
    normalized_query = normalize_query_text(query)
    if feature_cache is None:
        return {
            "proposal_backend": "uniform",
            "feature_cache_used": False,
            "query_raw": str(query or ""),
            "query_normalized": normalized_query,
            "query_source": str(query_source or "model"),
            "proposal_candidate_count": 0,
            "proposal_candidate_frame_indices": [],
            "proposal_candidate_frame_scores": [],
            "proposal_candidate_windows": [],
            "selected_frame_indices": [],
            "selected_frame_scores": [],
            "proposal_fallback_reason": "missing_feature_cache",
        }
    if proposal_runtime is None or not normalized_query:
        return {
            "proposal_backend": "uniform",
            "feature_cache_used": True,
            "query_raw": str(query or ""),
            "query_normalized": normalized_query,
            "query_source": str(query_source or "model"),
            "proposal_candidate_count": 0,
            "proposal_candidate_frame_indices": [],
            "proposal_candidate_frame_scores": [],
            "proposal_candidate_windows": [],
            "selected_frame_indices": [],
            "selected_frame_scores": [],
            "proposal_fallback_reason": "missing_query_encoder" if proposal_runtime is None else "empty_query",
        }

    embeddings = feature_cache.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return {
            "proposal_backend": "uniform",
            "feature_cache_used": True,
            "query_raw": str(query or ""),
            "query_normalized": normalized_query,
            "query_source": str(query_source or "model"),
            "proposal_candidate_count": 0,
            "proposal_candidate_frame_indices": [],
            "proposal_candidate_frame_scores": [],
            "proposal_candidate_windows": [],
            "selected_frame_indices": [],
            "selected_frame_scores": [],
            "proposal_fallback_reason": "invalid_feature_cache",
        }

    text_embeddings = proposal_runtime.encode_texts([normalized_query])
    if not isinstance(text_embeddings, torch.Tensor):
        raise TypeError("proposal_runtime.encode_texts must return a torch.Tensor.")
    if text_embeddings.ndim == 1:
        text_embeddings = text_embeddings.unsqueeze(0)
    frame_embeddings = embeddings.detach().float().cpu()
    text_embeddings = text_embeddings.detach().float().cpu()
    if not bool(feature_cache.get("normalized", False)):
        frame_embeddings = _l2_normalize(frame_embeddings)
    text_embeddings = _l2_normalize(text_embeddings)

    start_index = max(int(math.floor(float(start_sec) * float(fps))), 0)
    end_index = min(int(math.floor(float(end_sec) * float(fps))), int(frame_embeddings.shape[0]) - 1)
    if end_index < start_index:
        end_index = start_index
    search_embeddings = frame_embeddings[start_index : end_index + 1]
    scores = torch.matmul(search_embeddings, text_embeddings[0])
    effective_top_k = max(int(num_frames), min(int(top_k_candidates), int(scores.shape[0])))
    top_scores, top_local_indices = torch.topk(scores, k=effective_top_k)
    candidate_global_indices = [int(index) + start_index for index in top_local_indices.tolist()]
    candidate_frame_scores = [round(float(score), 6) for score in top_scores.tolist()]

    max_gap_frames = max(1, int(round(float(candidate_merge_gap_sec) * float(max(fps, 1e-6)))))
    candidate_windows: List[Dict[str, Any]] = []
    for cluster in _cluster_sorted_indices(candidate_global_indices, max_gap=max_gap_frames):
        cluster_scores = [float(scores[int(index - start_index)].item()) for index in cluster]
        candidate_windows.append(
            {
                "start_frame_index": int(cluster[0]),
                "end_frame_index": int(cluster[-1]),
                "start_sec": round(float(cluster[0]) / max(float(fps), 1e-6), 6),
                "end_sec": round(float(cluster[-1]) / max(float(fps), 1e-6), 6),
                "frame_indices": [int(index) for index in cluster],
                "score_max": round(max(cluster_scores), 6),
                "score_mean": round(sum(cluster_scores) / float(len(cluster_scores)), 6),
            }
        )
    candidate_windows.sort(key=lambda item: (-float(item["score_max"]), -float(item["score_mean"]), int(item["start_frame_index"])))

    if not candidate_windows:
        return {
            "proposal_backend": "uniform",
            "feature_cache_used": True,
            "query_raw": str(query or ""),
            "query_normalized": normalized_query,
            "query_source": str(query_source or "model"),
            "proposal_candidate_count": 0,
            "proposal_candidate_frame_indices": candidate_global_indices,
            "proposal_candidate_frame_scores": candidate_frame_scores,
            "proposal_candidate_windows": [],
            "selected_frame_indices": [],
            "selected_frame_scores": [],
            "proposal_fallback_reason": "empty_candidate_windows",
        }

    selected_window = candidate_windows[0]
    selected_indices_sorted = sorted(
        selected_window["frame_indices"],
        key=lambda index: float(scores[int(index - start_index)].item()),
        reverse=True,
    )[: max(1, int(num_frames))]
    selected_indices = sorted(int(index) for index in selected_indices_sorted)
    selected_scores = [round(float(scores[int(index - start_index)].item()), 6) for index in selected_indices]
    return {
        "proposal_backend": "feature_topk",
        "feature_cache_used": True,
        "query_raw": str(query or ""),
        "query_normalized": normalized_query,
        "query_source": str(query_source or "model"),
        "proposal_candidate_count": len(candidate_windows),
        "proposal_candidate_frame_indices": candidate_global_indices,
        "proposal_candidate_frame_scores": candidate_frame_scores,
        "proposal_candidate_windows": candidate_windows,
        "selected_frame_indices": selected_indices,
        "selected_frame_scores": selected_scores,
        "proposal_fallback_reason": None,
    }


@dataclass
class SiglipFeatureEncoder:
    model: Any
    processor: Any
    device: str = "cpu"
    model_name: str = ""

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        torch_dtype: str | torch.dtype = "auto",
        device: str = "cpu",
    ) -> "SiglipFeatureEncoder":
        from transformers import AutoModel, AutoProcessor

        dtype_arg = None
        if isinstance(torch_dtype, str):
            if torch_dtype not in {"", "auto", "none"}:
                dtype_arg = getattr(torch, torch_dtype)
        else:
            dtype_arg = torch_dtype
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, torch_dtype=dtype_arg, local_files_only=True)
        model.to(device)
        model.eval()
        return cls(model=model, processor=processor, device=device, model_name=model_path)

    def _to_pil_images(self, images: torch.Tensor) -> List[Any]:
        from PIL import Image

        pil_images: List[Any] = []
        for frame in images.detach().cpu():
            if frame.ndim != 3:
                raise ValueError("Expected image tensor in CHW format.")
            array = frame.permute(1, 2, 0).contiguous().numpy()
            if array.dtype != np.uint8:
                array = array.astype("float32")
                array = array.clip(0.0, 255.0).astype("uint8")
            pil_images.append(Image.fromarray(array))
        return pil_images

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        pil_images = self._to_pil_images(images)
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                features = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)
                features = outputs
        features = coerce_encoder_feature_tensor(
            features,
            preferred_keys=("image_embeds", "pooler_output", "last_hidden_state", "hidden_states"),
        )
        return _l2_normalize(features.detach().cpu())

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                features = self.model.get_text_features(**inputs)
            else:
                outputs = self.model(**inputs)
                features = outputs
        features = coerce_encoder_feature_tensor(
            features,
            preferred_keys=("text_embeds", "pooler_output", "last_hidden_state", "hidden_states"),
        )
        return _l2_normalize(features.detach().cpu())
