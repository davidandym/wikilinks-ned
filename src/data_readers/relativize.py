import os
import json

from data_readers.utils import Counter

"""Code for relativizing our embedding files.
Currently only does article embeddings, I think
"""

def get_vocab_and_wiki_id_count(data_dir_paths):
    word_counter = Counter()
    wiki_id_counter = Counter()
    count = 0
    success_count = 0
    for dir_path in data_dir_paths:
        print 'checking out directory', dir_path
        if os.path.isdir(dir_path):
            for _, fname in enumerate(os.listdir(dir_path)):
                print 'indexing file', fname
                f = open(os.path.join(dir_path, fname), 'r')
                lines = f.readlines()
                for line in lines:
                    count += 1
                    json_obj = json.loads(line)
                    if json_obj['wikiId'] != None:
                        success_count += 1
                        wiki_id_counter.increment_count(json_obj['wikiId'], 1)
                    if json_obj['right_context'] != None:
                        for word in json_obj['right_context']:
                            word_counter.increment_count(word, 1)
                    if json_obj['left_context'] != None:
                        for word in json_obj['left_context']:
                            word_counter.increment_count(word, 1)
                    if json_obj['mention_as_list'] != None:
                        for word in json_obj['mention_as_list']:
                            word_counter.increment_count(word, 1)
        else:
            print 'Invalid data dir', dir_path
    print 'total lines', count
    print 'total lines consumed', success_count
    return word_counter, wiki_id_counter

def relativize(file, outfile, word_counter):
    print 'relativizing', file, '->', outfile
    f = open(file)
    o = open(outfile, 'w')
    voc = Counter()
    count_failed = 0
    first=True
    for line in f:
        if first:
            first=False
            continue
        word = line[:line.find(' ')]
        if word_counter.get_count(word) > 0:
            #print "Keeping word vector for " + word
            voc.increment_count(word, 1)
            o.write(line)
        else:
            count_failed += 1
    for word in word_counter.keys():
        if word not in voc.keys():
            print "Missing " + repr(word) + " with count " + repr(word_counter.get_count(word))
    print 'final size of vocab:', len(voc)
    print 'got rid of', count_failed
    f.close()
    o.close()

if __name__=="__main__":
    data_dirs_to_index = [
        '/home/david/data/wikilinksNED_dataset/wikilinksNED_dataset/test',
        '/home/david/data/wikilinksNED_dataset/wikilinksNED_dataset/train',
        '/home/david/data/wikilinksNED_dataset/wikilinksNED_dataset/validation'
    ]

    word_count, wiki_count = get_vocab_and_wiki_id_count(data_dirs_to_index)
    print 'size of wiki_id vocab', len(wiki_count)
    print 'size of word vocab', len(word_count)

    relativize('data/glove.42B.300d.txt', '/home/david/data/wikilinks_embeddings/glove-word-relativized', word_count)
    #relativize('/Users/davidmueller/data/wikilink_embeddings/dim300context-vecs', '/Users/davidmueller/data/wikilink_embeddings/dim300context-vecs-relativized', wiki_count)

