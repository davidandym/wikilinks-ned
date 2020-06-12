import pickle
import sys

class CategoryParser(object):
    def __init__(self, path, lim=None):
        self._path = path
        self._num_pages = 0
        self._num_titles = 0
        self._num_ids = 0
        self._num_end_pages = 0
        self._num_categories = 0
        self._num_errs = 0
        self._errs = []
        self._page_title_db = dict()
        self._page_category_db = dict()
        self._category_count = dict()
        self._limit = lim

    def parse(self):
        curr_page_title = ""
        curr_page_id = 0
        curr_page_cats = []
        print 'beginning parsing of %s' % self._path
        with open(self._path, 'r') as wiki_dump:
            for line in wiki_dump:
                if line.find('<page>') != -1:
                    self._num_pages += 1
                    if self._num_pages % 1000 == 0:
                        print "%d pages seen so far" % self._num_pages
                elif line.find('<title>') != -1:
                    self._num_titles += 1
                    curr_page_title = line[line.find('<title>') + 7:line.find('</title>')]
                elif line.startswith('    <id>'):
                    self._num_ids += 1
                    page_id = line[line.find('<id>') + 4:line.find('</id>')]
                    curr_page_id = int(page_id)
                elif line.startswith('[[Category:'):
                    self._num_categories += 1
                    category = line[11:line.find(']]')]
                    curr_page_cats.append(category)
                    if self._category_count.has_key(category):
                        self._category_count[category] += 1
                    else:
                        self._category_count[category] = 1
                elif line.find('</page>') != -1:
                    self._num_end_pages += 1
                    if self._page_title_db.has_key(curr_page_id):
                        self._num_errs += 1
                        self._errs.append(curr_page_id)
                    else:
                        self._page_title_db[curr_page_id] = curr_page_title

                    if self._page_category_db.has_key(curr_page_id):
                        self._num_errs += 1
                        self._errs.append(curr_page_id)
                    else:
                        self._page_category_db[curr_page_id] = curr_page_cats

                    if self._limit is not None and self._num_end_pages > self._limit:
                        break
                    curr_page_id = 0
                    curr_page_title = ""
                    curr_page_cats = []
            print 'parsing complete'
            print '%d pages' % self._num_pages
            print '%d ids' % self._num_ids
            print '%d titles' % self._num_titles
            print '%d end pages' % self._num_end_pages
            print '%d num errors' % self._num_errs

if __name__ == '__main__':
    filename = sys.argv[1]
    if len(sys.argv) > 2:
        limit = sys.argv[2]
        dump_file = 'data/category_obj_%s.pickle' % limit
    else:
        limit = None
        dump_file = 'data/category_obj.pickle'
    parser = CategoryParser(filename, lim=limit)
    parser.parse()
    pickle.dump(parser, open(dump_file, 'w'))
