import pickle
import json
from CategoryParser import CategoryParser

def pull_out_title_db():
    category_obj = pickle.load(open('data/category_obj.pickle', 'r'))
    json.dump(category_obj._page_title_db, open('data/page_title.json', 'w'))

if __name__ == '__main__':
    pull_out_title_db()
