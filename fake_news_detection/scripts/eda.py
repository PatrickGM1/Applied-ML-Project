from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LABELED_DIR = PROJECT_DIR / 'data' / 'processed' / 'labeled'
CLEANED_DIR = PROJECT_DIR / 'data' / 'processed' / 'cleaned_text'
EDA_DIR = PROJECT_DIR / 'artifacts' / 'eda'

LABEL_ORDER = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
BINARY_ORDER = ['fake', 'real']

HIST_COLUMNS = ['hist1', 'hist2', 'hist3', 'hist4', 'hist5']
HIST_LABELS = [
    'Barely-true count',
    'False count',
    'Half-true count',
    'Mostly-true count',
    'Pants-on-fire count',
]


def load_train():
    return pd.read_csv(LABELED_DIR / 'train.processed.csv')


def load_binary_train():
    return pd.read_csv(LABELED_DIR / 'train_binary.processed.csv')


def save_figure(file_name):
    plt.tight_layout()
    plt.savefig(EDA_DIR / file_name, dpi=150)
    plt.close()


def plot_label_distribution(frame):
    counts = (
        frame['label']
        .value_counts()
        .reindex(LABEL_ORDER)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(counts.index, counts.values, color=sns.color_palette('Set2', len(LABEL_ORDER)))
    axes[0].set_title('6-Class Label Distribution (train)')
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=20)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 10, str(v), ha='center', fontsize=9)

    binary_counts = (
        frame['label_binary']
        .dropna()
        .value_counts()
        .reindex(BINARY_ORDER)
    )
    axes[1].bar(binary_counts.index, binary_counts.values, color=sns.color_palette('Set1', 2))
    axes[1].set_title('Binary Label Distribution (train)')
    axes[1].set_xlabel('Label')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(binary_counts.values):
        axes[1].text(i, v + 10, str(v), ha='center', fontsize=9)

    save_figure('label_distribution.png')


def plot_missing_values(frame):
    missing = frame.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(missing.index, missing.values, color=sns.color_palette('Oranges_r', len(missing)))
    ax.set_title('Missing Values per Column (train)')
    ax.set_xlabel('Missing count')
    for i, v in enumerate(missing.values):
        ax.text(v + 5, i, str(v), va='center', fontsize=9)

    save_figure('missing_values.png')


def plot_text_length_distribution(frame):
    frame = frame.copy()
    frame['word_count'] = frame['statement'].dropna().apply(lambda t: len(str(t).split()))

    fig, ax = plt.subplots(figsize=(12, 5))
    for label in LABEL_ORDER:
        subset = frame[frame['label'] == label]['word_count'].dropna()
        ax.hist(subset, bins=40, alpha=0.5, label=label)

    ax.set_title('Statement Word Count Distribution by Label (train)')
    ax.set_xlabel('Word count')
    ax.set_ylabel('Frequency')
    ax.legend()
    save_figure('text_length_distribution.png')


def plot_fake_rate_by_party(frame):
    binary = frame[frame['label_binary'].notna()].copy()
    party_counts = binary['party'].value_counts()
    top_parties = party_counts[party_counts >= 50].index.tolist()

    subset = binary[binary['party'].isin(top_parties)].copy()
    fake_rate = (
        subset.groupby('party')['label_binary']
        .apply(lambda s: (s == 'fake').mean())
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(fake_rate.index, fake_rate.values, color=sns.color_palette('RdYlGn_r', len(fake_rate)))
    ax.set_title('Fake Rate by Party (parties with ≥50 statements, train)')
    ax.set_xlabel('Party')
    ax.set_ylabel('Proportion labelled fake')
    ax.set_ylim(0, 1)
    ax.axhline(0.5, linestyle='--', color='black', linewidth=0.8, label='50% line')
    ax.legend()
    ax.tick_params(axis='x', rotation=30)
    save_figure('fake_rate_by_party.png')


def plot_tsne(frame, sample_size=2000, random_state=42):
    """t-SNE projection of TF-IDF text embeddings coloured by 6-class label."""
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(frame), size=min(sample_size, len(frame)), replace=False)
    sample = frame.iloc[idx].copy()

    text_col = 'statement_clean' if 'statement_clean' in sample.columns else 'statement'
    texts = sample[text_col].fillna('').tolist()

    vectorizer = TfidfVectorizer(max_features=500, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()

    tsne = TSNE(n_components=2, perplexity=30, random_state=random_state, max_iter=1000)
    coords = tsne.fit_transform(tfidf_matrix)

    palette = sns.color_palette('tab10', len(LABEL_ORDER))
    label_to_color = {label: palette[i] for i, label in enumerate(LABEL_ORDER)}

    fig, ax = plt.subplots(figsize=(10, 7))
    for label in LABEL_ORDER:
        mask = sample['label'] == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[label_to_color[label]],
            label=label,
            alpha=0.6,
            s=15,
        )
    ax.set_title(f't-SNE of TF-IDF Text Embeddings (n={len(sample)}, train)')
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.legend(title='Label', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    save_figure('tsne_text_embeddings.png')


def plot_history_counts(frame):
    frame = frame.copy()
    for col in HIST_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')

    fig, axes = plt.subplots(1, len(HIST_COLUMNS), figsize=(18, 4), sharey=False)

    for ax, col, label_text in zip(axes, HIST_COLUMNS, HIST_LABELS):
        fake_vals = frame[frame['label_binary'] == 'fake'][col].dropna()
        real_vals = frame[frame['label_binary'] == 'real'][col].dropna()
        ax.hist(fake_vals, bins=30, alpha=0.6, label='fake', color='tomato')
        ax.hist(real_vals, bins=30, alpha=0.6, label='real', color='steelblue')
        ax.set_title(label_text, fontsize=9)
        ax.set_xlabel('Count')
        ax.legend(fontsize=7)

    fig.suptitle('Speaker History Counts: Fake vs Real (train binary subset)', y=1.02)
    plt.savefig(EDA_DIR / 'history_counts.png', dpi=150, bbox_inches='tight')
    plt.close()


def summary_lines(frame):
    missing = frame.isnull().sum()
    word_counts = frame['statement'].dropna().apply(lambda t: len(str(t).split()))

    lines = [
        'Dataset summary (train)',
        f'rows: {len(frame)}',
        '',
        'label counts:',
        frame['label'].value_counts().to_string(),
        '',
        'binary label counts:',
        frame['label_binary'].value_counts(dropna=False).to_string(),
        '',
        'missing values:',
        missing[missing > 0].to_string(),
        '',
        'statement word count stats:',
        word_counts.describe().to_string(),
    ]
    return lines


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_train()
    binary_frame = load_binary_train()

    # Load cleaned text version if available (has statement_clean column)
    cleaned_path = CLEANED_DIR / 'train.processed.csv'
    tsne_frame = pd.read_csv(cleaned_path) if cleaned_path.exists() else frame

    summary_path = EDA_DIR / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write('\n'.join(summary_lines(frame)))

    plot_label_distribution(frame)
    plot_missing_values(frame)
    plot_text_length_distribution(frame)
    plot_fake_rate_by_party(frame)
    plot_history_counts(binary_frame)
    plot_tsne(tsne_frame)

    print(f'EDA done. Plots and summary saved in: {EDA_DIR}')


if __name__ == '__main__':
    main()
