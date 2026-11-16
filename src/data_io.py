import os
import pandas as pd
from collections import Counter

TRAIN_PATH = "data/sentimix/Semeval_2020_task9_data/Hinglish/Hinglish_train_14k_split_conll.txt"
DEV_PATH = "data/sentimix/Semeval_2020_task9_data/Hinglish/Hinglish_dev_3k_split_conll.txt"


def parse_conll_file(file_path: str) -> list[dict]:
    """Parse SentiMix CoNLL file into list of tweet dicts."""
    tweets = []
    current_tweet = None

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                if current_tweet and current_tweet["tokens"]:
                    tweets.append(current_tweet)
                    current_tweet = None
                continue

            parts = line.split("\t")
            if parts[0] == "meta":
                if current_tweet and current_tweet["tokens"]:
                    tweets.append(current_tweet)
                uid = parts[1] if len(parts) > 1 else ""
                sentiment = parts[2] if len(parts) > 2 else "neutral"
                current_tweet = {
                    "uid": uid,
                    "sentiment": sentiment,
                    "tokens": [],
                    "tags": [],
                }
            elif current_tweet is not None and len(parts) >= 2:
                current_tweet["tokens"].append(parts[0])
                current_tweet["tags"].append(parts[1])

    if current_tweet and current_tweet["tokens"]:
        tweets.append(current_tweet)

    return tweets


def load_dataset(data_dir: str = "data/sentimix") -> tuple[list[dict], list[dict]]:
    """Load train and dev splits."""
    train_file = os.path.join(
        data_dir, "Semeval_2020_task9_data/Hinglish/Hinglish_train_14k_split_conll.txt"
    )
    dev_file = os.path.join(
        data_dir, "Semeval_2020_task9_data/Hinglish/Hinglish_dev_3k_split_conll.txt"
    )
    return parse_conll_file(train_file), parse_conll_file(dev_file)


def summarize_split(split_name: str, tweets: list[dict]) -> list[dict]:
    """Return summary records for a split."""
    n_tweets = len(tweets)
    n_tokens = sum(len(t["tokens"]) for t in tweets)
    sent_counts = Counter(t["sentiment"] for t in tweets)
    tag_counts = Counter(tag for t in tweets for tag in t["tags"])

    records = [
        {"split": split_name, "category": "overall", "item": "tweet_count", "count": n_tweets},
        {"split": split_name, "category": "overall", "item": "token_count", "count": n_tokens},
    ]
    for sent, cnt in sorted(sent_counts.items()):
        records.append({"split": split_name, "category": "sentiment", "item": sent, "count": cnt})
    for tag, cnt in sorted(tag_counts.items()):
        records.append({"split": split_name, "category": "language_tag", "item": tag, "count": cnt})
    return records


def run_data_summary(output_path: str = "results/data_summary.csv") -> pd.DataFrame:
    """Generate and save data summary table."""
    train_tweets, dev_tweets = load_dataset()
    records = summarize_split("train", train_tweets) + summarize_split("dev", dev_tweets)
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_data_summary()
