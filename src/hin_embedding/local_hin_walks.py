
from src.utils.utils import *
import random
import pandas as pd

# 数据路径
DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v3"
RWALK_DIR = "../../datas/hin_embedding"
PRE_DATA_DIR = "../../datas/pre_data"
ADJ_DIR = "../../datas/adj"

valid_author_name_ids = pd.read_pickle(PRE_DATA_DIR+'/valid_author_name_ids.pkl')
valid_author_name_paper_ids = pd.read_pickle(PRE_DATA_DIR+'/valid_author_name_paper_ids.pkl')

def load_datas(path):
    results = {}
    edges_unordered = np.genfromtxt(path, dtype=np.dtype(str))
    for i in edges_unordered:
        if i[0] not in results:
            results[i[0]] = []
        results[i[0]].append(i[1])
    return results

def rand_walks(name,papers,walks_num):

    # abstract = load_datas(ADJ_DIR + "/abstract/" + name + "/pubs_network.txt")
    author_orgs = load_datas(ADJ_DIR + "/author_org/" + name + "/pubs_network.txt")
    authors = load_datas(ADJ_DIR + "/authors/" + name + "/pubs_network.txt")
    # keywords = load_datas(ADJ_DIR + "/keywords/" + name + "/pubs_network.txt")
    # title = load_datas(ADJ_DIR + "/title/" + name + "/pubs_network.txt")
    venue = load_datas(ADJ_DIR + "/venue/" + name + "/pubs_network.txt")

    out_path = RWALK_DIR+"/"+name
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    outfile = open(out_path+"/rand_walks.txt", 'w')

    for paper_author in papers:
        for paper in paper_author:
            walks_list = ""
            for i in range(walks_num):
                if paper in authors.keys():
                    papers = authors[paper]
                    numw = len(papers)
                    next_paper_id = random.randrange(numw)
                    paper = papers[next_paper_id]
                    walks_list += " " + paper
                # if paper in abstract.keys():
                #     papers = abstract[paper]
                #     numw = len(papers)
                #     next_paper_id = random.randrange(numw)
                #     paper = papers[next_paper_id]
                #     walks_list += " " + paper
                if paper in author_orgs.keys():
                    papers = author_orgs[paper]
                    numw = len(papers)
                    next_paper_id = random.randrange(numw)
                    paper = papers[next_paper_id]
                    walks_list += " " + paper
                if paper in venue.keys():
                    papers = venue[paper]
                    numw = len(papers)
                    next_paper_id = random.randrange(numw)
                    paper = papers[next_paper_id]
                    walks_list += " " + paper
                # if paper in keywords.keys():
                #     papers = keywords[paper]
                #     numw = len(papers)
                #     next_paper_id = random.randrange(numw)
                #     paper = papers[next_paper_id]
                #     walks_list += " " + paper
                # if paper in title.keys():
                #     papers = title[paper]
                #     numw = len(papers)
                #     next_paper_id = random.randrange(numw)
                #     paper = papers[next_paper_id]
                #     walks_list += " " + paper
            outfile.write(walks_list + "\n")
            print(walks_list)
    outfile.close()

def convert(name):
    rrrr = '[!“”"#$%&\'()*+,-./:)‘′(;<=>?@[\\]^-_` •！@#￥&*（）——+~}】【|、|？《》，。：“；‘{|}~—～’*《》<>「」{}【】()/\\\[\] ]+'
    stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the']
    name = name.lower()
    name = re.sub(rrrr, ' ', name.strip().lower())
    name = re.sub(r'\s{2,}', '', name).strip()
    name = re.sub(r'[0-9]', '', name).strip()
    new_name = ""
    for word in name.split():
        if word not in stopword:
            new_name +=word
    return new_name


def main():

    for index,name in enumerate(valid_author_name_ids["author_name"]):
        name = convert(name)
        print(index,name)
        papers = []
        for author in valid_author_name_ids.iloc[index]["author_ids"]:
            paper = valid_author_name_paper_ids[valid_author_name_paper_ids["author_id"] == author]["paper_ids"]
            paper = paper.values.tolist()[0]
            papers.append(paper)
        if index>=0:
            rand_walks(name,papers,5)



if __name__ == '__main__':
    main()
