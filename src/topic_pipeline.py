"""Re-run original LDA preprocessing and a separate train-only topic experiment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel, Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from event_pipeline import END, ROOT, SPLIT, load_tweets


def tokenize(text: str) -> list[str]:
    """Preserve the original URL/mention removal and hashtag token handling."""
    cleaned = re.sub(r"http\S+|@\w+", " ", text.lower()).replace("#", " ")
    return simple_preprocess(cleaned, deacc=True, min_len=3)


def preprocess(
    train: list[list[str]], other: list[list[str]]
) -> tuple[list[list[str]], list[list[str]]]:
    """Fit bigrams on training text only; apply identical stemming to both sets."""
    phrases = Phraser(Phrases(train, min_count=5, threshold=10))
    stop = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    def transform(documents: list[list[str]]) -> list[list[str]]:
        return [
            [stemmer.stem(w) for w in phrases[doc] if w not in stop]
            for doc in documents
        ]

    return transform(train), transform(other)


def fit_model(tokens: list[list[str]], topics: int) -> tuple[LdaModel, list]:
    """Fit the original seed/passes/priors without filtering the dictionary."""
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(doc) for doc in tokens]
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=topics,
        random_state=42,
        passes=10,
        chunksize=2000,
        alpha="auto",
        eta="auto",
    )
    return model, corpus


def run(output: Path) -> None:
    """Keep historical replication and prospective preprocessing clearly separate."""
    output.mkdir(parents=True, exist_ok=True)
    tweets = load_tweets(ROOT / "data/raw")
    tweets = tweets.loc[
        (tweets.time >= pd.Timestamp("2016-11-08", tz="UTC")) & (tweets.time < END)
    ].reset_index(drop=True)
    raw_tokens = [tokenize(t) for t in tweets.text]
    all_tokens, _ = preprocess(raw_tokens, [])
    keep = np.array([bool(t) for t in all_tokens])
    full_tokens = [t for t in all_tokens if t]
    legacy, corpus = fit_model(full_tokens, 20)
    probabilities = [
        dict(legacy.get_document_topics(bow, minimum_probability=0))[0]
        for bow in corpus
    ]
    replicated = (
        pd.DataFrame(
            {
                "datetime": tweets.loc[keep, "time"].dt.floor("min"),
                "replicated": probabilities,
            }
        )
        .groupby("datetime")
        .max()
    )
    existing = pd.read_csv(ROOT / "data/processed/reg_input_topic0_minute.csv")
    existing["datetime"] = pd.to_datetime(existing.datetime, utc=True)
    joined = existing.merge(replicated, on="datetime")
    legacy_report = {
        "documents": len(full_tokens),
        "seed": 42,
        "passes": 10,
        "coherence_cv": CoherenceModel(
            model=legacy,
            texts=full_tokens,
            dictionary=legacy.id2word,
            coherence="c_v",
            processes=1,
        ).get_coherence(),
        "topic0_correlation_existing": joined.topic0.corr(joined.replicated),
        "topic0_mae_existing": float((joined.topic0 - joined.replicated).abs().mean()),
        "compared_minutes": len(joined),
        "topics": {str(i): legacy.show_topic(i, 12) for i in range(20)},
    }
    (output / "legacy_lda.json").write_text(
        json.dumps(legacy_report, indent=2, default=float)
    )
    train_mask = (tweets.time < SPLIT).to_numpy()
    train_tokens, test_tokens = preprocess(
        [doc for doc, flag in zip(raw_tokens, train_mask, strict=True) if flag],
        [doc for doc, flag in zip(raw_tokens, train_mask, strict=True) if not flag],
    )
    train_tokens = [doc for doc in train_tokens if doc]
    rows = []
    chosen_model = None
    for k in (10, 20, 30):
        model, _ = fit_model(train_tokens, k)
        score = CoherenceModel(
            model=model,
            texts=train_tokens,
            dictionary=model.id2word,
            coherence="c_v",
            processes=1,
        ).get_coherence()
        rows.append(
            {
                "topics": k,
                "train_coherence_cv": score,
                "vocabulary": len(model.id2word),
                "seed": 42,
            }
        )
        if chosen_model is None or score > max(
            r["train_coherence_cv"] for r in rows[:-1]
        ):
            chosen_model = model
    pd.DataFrame(rows).to_csv(output / "topic_count_comparison.csv", index=False)
    # Label a trade-related topic by lexical mass, never by a market outcome.
    terms = [
        i
        for word, i in chosen_model.id2word.token2id.items()
        if word in {"tariff", "china", "trade"}
    ]
    if not terms:
        raise ValueError("Trade lexicon absent from training vocabulary")
    trade_topic = int(chosen_model.get_topics()[:, terms].sum(axis=1).argmax())
    heldout_frame = tweets.loc[~train_mask].copy()
    heldout_frame["topic0"] = [
        dict(
            chosen_model.get_document_topics(
                chosen_model.id2word.doc2bow(doc), minimum_probability=0
            )
        )[trade_topic]
        for doc in test_tokens
    ]
    train_raw = [doc for doc, flag in zip(raw_tokens, train_mask, strict=True) if flag]
    train_transformed, _ = preprocess(train_raw, [])
    train_frame = tweets.loc[train_mask].copy()
    train_frame["topic0"] = [
        dict(
            chosen_model.get_document_topics(
                chosen_model.id2word.doc2bow(doc), minimum_probability=0
            )
        )[trade_topic]
        for doc in train_transformed
    ]
    scores = pd.concat([train_frame, heldout_frame])
    scores["datetime"] = scores.time.dt.floor("min")
    scores.groupby("datetime").topic0.max().to_csv(
        output / "train_only_topic_scores.csv"
    )
    chosen_model.save(str(output / "train_only_lda.model"))
    (output / "train_only_lda.json").write_text(
        json.dumps(
            {
                "split_utc": str(SPLIT),
                "train_documents": len(train_frame),
                "test_documents": len(heldout_frame),
                "selected_k": chosen_model.num_topics,
                "trade_topic_id": trade_topic,
                "trade_topic_words": chosen_model.show_topic(trade_topic, 15),
                "selection": "maximum training c_v, seed 42, k=10/20/30",
                "limitations": [
                    "One seed; topic count stability not established",
                    "Retrospective split is not an untouched future test",
                    "Model artifact is local-only pending source rights review",
                ],
            },
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
