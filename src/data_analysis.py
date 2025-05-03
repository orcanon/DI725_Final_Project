import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import matplotlib
matplotlib.use('Agg')

# Load the dataset
captions_df = pd.read_csv('../data/RISCM/captions.csv')

# Concatenate all captions into a single series
captions_all = pd.concat([captions_df[f'caption_{i}'] for i in range(1, 6)])

# Analyze caption lengths
caption_lengths = captions_all.apply(lambda x: len(x.split()))

# Plot caption length distribution
plt.figure(figsize=(10, 5))
sns.histplot(caption_lengths, bins=30, kde=True)
plt.title('Caption Length Distribution')
plt.xlabel('Caption Length (words)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('../figures/caption_length_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

# Vocabulary analysis
words = captions_all.apply(lambda x: re.findall(r'\b\w+\b', x.lower()))
vocabulary = Counter(word for caption in words for word in caption)

# Top 20 most frequent words
most_common_words = vocabulary.most_common(20)
words_df = pd.DataFrame(most_common_words, columns=['word', 'count'])

# Plot most frequent words
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='word', data=words_df, palette='viridis')
plt.title('Top 20 Most Frequent Words')
plt.xlabel('Occurrences')
plt.ylabel('Words')
plt.grid(True)
plt.savefig('../figures/top_20_most_frequent_words.png')
plt.close()

# Dataset split analysis
split_counts = captions_df['split'].value_counts()

# Plot dataset splits
plt.figure(figsize=(6, 4))
sns.barplot(x=split_counts.index, y=split_counts.values, palette='pastel')
plt.title('Dataset Splits')
plt.xlabel('Split')
plt.ylabel('Number of Images')
plt.grid(True)
plt.savefig('../figures/dataset_splits.png')
plt.close()

# Source analysis
source_counts = captions_df['source'].value_counts()

# Plot data sources
plt.figure(figsize=(8, 4))
sns.barplot(x=source_counts.index, y=source_counts.values, palette='mako')
plt.title('Dataset Sources')
plt.xlabel('Source')
plt.ylabel('Number of Images')
plt.grid(True)
plt.savefig('../figures/dataset_sources.png')
plt.close()
