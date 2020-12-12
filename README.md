# fake-news-detection

# Final Year Project

    - /
            -- credibility.py : generates dictionary of authors / speakers with credibility scores and stores object on disk
            -- model.py : generates word embeddings using Word2Vec model after tokenization and data cleaning
            -- config.py : stores global tuning parameters
            -- tfidf.py : generates and stores traditional tfidf vectors
            -- classifiers.py : formats train and test data, trains model and stores the results obtained on disk

    - data /
            -- vectors.tsv : output of word2vec model, i.e generated word vectors
            -- metadata.tsv : vocabulary corresponding to word2vec model
            -- labels.tsv : labels of data tuples after aggregation
            -- input_vectors_words.tsv : input tuples to classifiers with words
            -- input_vectors.tsv : input tuples to classifiers in the form of vector
            -- tfidf.tsv : tfidf vectors correspoding to statements in datasets
            -- credibility_lookup.pickle : serialized credibility lookup
            -- output_tfidf.json : results of classifier with tfidf
            -- output_word2vec.json : results of classifier with word embeddings
