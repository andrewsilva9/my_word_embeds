# my_word_embeds

This documentation is mostly all out of date. For now, check out http://andrew-silva.com/blog.html, which will slowly be updated with more up-to-date documentation on this (and other) repos.

~~Now, with `masters` full of .txt files, run:~~

```python create_seq_pairs.py```

~~Note that I'm casting all text to lowercase for simplicity, and removing some punctuation. After that cleaning up, this creates pairs of "Friend text" and "My response" in `pairs.pkl`.~~

~~After that, your text is ready to go and you can run~~

```python my_cbow_embeds.py```

~~Which will being training your word embedding model. Currently, I have it set to train for 100 epochs, you can change this to be whatever you think is appropriate. There is some commented out code that lets you resume training from a saved checkpoint, which you can use if your training cuts out for any reason. Note that this will flood your machine with models (it saves 1 every epoch), so you might want to clean those out. In the future I'll try to make it a bit neater so that you can load models in when you launch the training.~~

~~After getting your embedding model, use the `word_to_ix.pkl` file you have to map words to embeddings. Like so: ~~

```python create_mapping.py```

~~This leaves you with a pickled dictionary that maps from word -> embedding.  Note that these are not stripped of punctuation, they are all cast to lowercase, they count typos as entirely separate words, etc. Basically the preprocessing step (make_text_pkl.py) could be improved (among other things).~~
