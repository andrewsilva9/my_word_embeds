# my_word_embeds

First, get your text corpus. I gathered all of my iMessages up using baskup: https://github.com/PeterKaminski09/baskup . I made a couple of changes to that script, namely I removed the "Me:" and "Friend:" for generating my word embeddings text. Because I don't want to learn "Me:" or "Friend:" as being the heads of every sentence...

Otherwise, within the baskup directory just run :
```
$ baskup.sh
$ cat*/* > master.txt
```
Which will put all of your text history into a file called 'master.txt'.

Move master.txt into this directory, and run 'make_text_pkl.py' in order to generate your raw text pickle, which the embedding model will use.

```python make_text_pkl.py```

Note that I'm casting all text to lowercase for simplicity.

After that, your text is ready to go and you can run

```python my_cbow_embeds.py```

Which will being training your word embedding model. Currently, I have it set to train for 100 epochs, you can change this to be whatever you think is appropriate. There is some commented out code that lets you resume training from a saved checkpoint, which you can use if your training cuts out for any reason. Note that this will flood your machine with models (it saves 1 every epoch), so you might want to clean those out. In the future I'll try to make it a bit neater so that you can load models in when you launch the training.

After getting your embedding model, use the `word_to_ix.pkl` file you have to map words to embeddings. Like so: 

```python create_mapping.py```

This leaves you with a pickled dictionary that maps from word -> embedding.  Note that these are not stripped of punctuation, they are all cast to lowercase, they count typos as entirely separate words, etc. Basically the preprocessing step (make_text_pkl.py) could be improved (among other things). 
