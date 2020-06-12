# wikilinks-ned


NED system for the paper [Effective Use of Context in Noisy Entity Linking](https://www.aclweb.org/anthology/D18-1126),
build for the [Wikilinks-NED dataset](https://www.aclweb.org/anthology/K17-1008).

### Requirements

Packages used are details in ``requirements.txt``.

Unfortunately, this code is really old and requires ``Python 2.7.17`` or similar, which I'm terribly sorry about :(

### Data

Running our system requires:

1. The **Wikilinks-NED dataset**
2. Word and Entity **embeddings**
3. A **page-title json** of wikipedia page ID to title strings (for title CNNs and feature construction)

The **page-title json** was built using a CategoryParser (in ``src/data_readers/CategoryParser.py``), which parses a variety of information
from a Wikipedia pages-articles dump and saves it into one large object. Then the script ``src/data_readers/pull_out_title_db.py`` simply saves the specific page ID to title string json that exists in that CategoryParser object to a separate json file.

There is certainly a much easier way to do this, it's relatively straightforward to do with a variety of Wiki-dump parsers.
My method is a by-product of the time when I was exploring category strings and embeddings as additional features for out system.

There are a couple of ways to get the **dataset and embeddings**:

- The original repo for creating the dataset and embeddings is [here](https://github.com/yotam-happy/NEDforNoisyText).
I found that I had to make some modifications to it to get it to work, which can be found [here](https://github.com/davidandym/NEDforNoisyText),
but note that my version _only_ contains scripts necessary for downloading and creating the dataset and embeddings, and not their actual NED system.
- Alternatively, you can email me at ``dam@jhu.edu`` and I can give you my dropbox link for both dataset and embeddings.

The word embedding size can be quite huge, so I suggest relativizing it. I have a script that does that (but it has hard-coded paths) at
``src/data_readers/relativize.py``.

### Config

Our system runs off of a configuration file which contains pointers to data files, training parameters, and model parameters.

An example configuration file can be seen below (these are all available options).
This configuration file builds a model with character-level CNN encoders, no hand-built features.
It runs for 5 epochs with a batch size of 100 and 4 negative entity samples per example.


```
{
	"files": {
	    "base-data-dir": "/exp/dmueller/wikilinks/data/",
	    "wikilink-stats": "wikilinksNED-train-stats",
	    "word-embeds": "dim300vecs-relativized",
	    "ent-embeds": "dim300context-vecs-relativized",
	    "wiki-page-titles": "page_title.json"
	},
	"training": {
	    "neg_samples": 4,
	    "neg_sample_from_cands": false,
	    "batch_size": 100,
	    "epochs": 5
	},
	"model": {
	    "features": false,
	    "character_cnn": true,
	    "ccnn_window_size": 5,
	    "lr": 0.001
	}
}
```

### Running our system

The entry point of the system is ``src/main.py``. It requires arguments ``--config_file``, which should point to a configuration file, and ``--experiment_dir``, which is the directory to save models to.

Running the system then looks like:
```
python src/main.py --config_file $CONFIG --experiment_dir $OUTPUT_DIRECTORY
```

### Paper

The paper describing this system is [Effective Use of Context in Noisy Entity Linking](https://www.aclweb.org/anthology/D18-1126), published at EMNLP 2018.

To cite this work, please use
```
@InProceedings{D18-1126,
  author = 	"Mueller, David and Durrett, Greg",
  title = 	"Effective Use of Context in Noisy Entity Linking",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"1024--1029",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1126"
}
```