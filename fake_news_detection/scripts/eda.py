from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LABELED_DIR = PROJECT_DIR / 'data' / 'processed' / 'labeled'
EDA_DIR = PROJECT_DIR / 'artifacts' / 'eda'

LABEL_ORDER = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
BINARY_ORDER = ['fake', 'real']

HIST_COLUMNS = ['hist1', 'hist2', 'hist3', 'hist4', 'hist5']
HIST_LABELS = [
    'Pants-fire count',
    'False count',
    'Barely-true count',
    'Half-true count',
    'Mostly-true count',
]


def load_train():
    return pd.read_csv(LABELED_DIR / 'train.processed.csv')


def load_binary_train():
    return pd.read_csv(LABELED_DIR / 'train_binary.processed.csv')


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

    plt.tight_layout()
    plt.savefig(EDA_DIR / 'label_distribution.png', dpi=150)
    plt.close()
    print('Saved: label_distribution.png')


def plot_missing_values(frame):
    missing = frame.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print('No missing values found.')
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(missing.index, missing.values, color=sns.color_palette('Oranges_r', len(missing)))
    ax.set_title('Missing Values per Column (train)')
    ax.set_xlabel('Missing count')
    for i, v in enumerate(missing.values):
        ax.text(v + 5, i, str(v), va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(EDA_DIR / 'missing_values.png', dpi=150)
    plt.close()
    print('Saved: missing_values.png')


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
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'text_length_distribution.png', dpi=150)
    plt.close()
    print('Saved: text_length_distribution.png')


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
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'fake_rate_by_party.png', dpi=150)
    plt.close()
    print('Saved: fake_rate_by_party.png')


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
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'history_counts.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: history_counts.png')


def print_summary(frame):
    print(f'\n=== Dataset Summary (train) ===')
    print(f'Rows: {len(frame)}')
    print(f'\nLabel counts:')
    print(frame['label'].value_counts().to_string())
    print(f'\nBinary label counts (mapped subset):')
    print(frame['label_binary'].value_counts(dropna=False).to_string())
    print(f'\nMissing values:')
    missing = frame.isnull().sum()
    print(missing[missing > 0].to_string())
    print(f'\nStatement word count stats:')
    word_counts = frame['statement'].dropna().apply(lambda t: len(str(t).split()))
    print(word_counts.describe().to_string())


def main():
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_train()
    binary_frame = load_binary_train()

    print_summary(frame)

    plot_label_distribution(frame)
    plot_missing_values(frame)
    plot_text_length_distribution(frame)
    plot_fake_rate_by_party(frame)
    plot_history_counts(binary_frame)

    print(f'\nAll plots saved to: {EDA_DIR}')


if __name__ == '__main__':
    main()
