import pandas as pd
from collections import defaultdict
import math
import pickle
import json 
import collections
import pandas as pd
import numpy as np


def obo_file_to_dict(filename):
    ONLY_ONE_ALLOWED_PER_STANZA = set(["id", "name", "def", "comment"])
    unique_tags = set([])

    current_type = None
    current_dict = None
    obo_dict = collections.OrderedDict()
    with open(filename) as lines: 
  
        for line in lines:
        
            #ignore the information from the head of the file
            if line.startswith("["):
                current_type = line.strip("[]\n")
                continue
            if current_type != "Term":
                continue
        
            # remove new-line character and comments    
            line = line.strip().split("!")[0]
            if len(line) == 0:
                continue
            
            #take line and divide into tag and value
            line = line.split(": ")
            tag = line[0]
            value = line[1]
        
            unique_tags.add(tag)
        
            #create new record for the new GO term
            if tag == "id":
                current_record = collections.defaultdict(list)
                obo_dict[value] = current_record
            
            if tag in current_record and tag in ONLY_ONE_ALLOWED_PER_STANZA:
                raise ValueError("more than one '%s' found in '%s' " % (tag, ", ".join([current_record[tag], value])) )
        
            current_record[tag].append(value)
            
    return obo_dict, unique_tags


def obo_dict_to_pandas(obo_dict, unique_tags):
    obo_panda = pd.DataFrame(columns = list(unique_tags))
    list_of_rows = []
    
    for key, dicto in obo_dict.items():
        new_row = pd.DataFrame([dicto])
        list_of_rows.append(new_row)
    
    obo_panda = pd.concat(list_of_rows, axis=0)    
    
    return obo_panda

def obo_csv_trim(csv_path = 'go-basic.obo.csv', onto_type='go'):
    

    go_terms = pd.read_csv(csv_path)

    #get only those GO terms that are not obsolete (not good anymore)
    valid_go_terms = go_terms.loc[go_terms['is_obsolete'].isna() ]  # 47278 x 19 --> 44261 x 19
    if onto_type == 'go':
        # selecting only those relationships mentioned in the paper
        terms_for_node2vec = valid_go_terms[["id", "is_a", "relationship", "namespace"]]
        terms_for_node2vec['id'] = terms_for_node2vec['id'].apply(lambda x: x.strip("['']")) 
        terms_for_node2vec['is_a'] = terms_for_node2vec['is_a'].apply(lambda x:  x.strip("[']").replace(' ', '').split("','") if type(x) is str else x) 
        terms_for_node2vec['relationship'] = terms_for_node2vec['relationship'].apply(lambda x:  x.strip("[]").split(", ") if type(x) is str else x) 
        
        terms_for_node2vec['namespace'] = terms_for_node2vec['namespace'].apply(lambda x:  x.strip("['']") if type(x) is str else x) 
        terms_for_node2vec.reset_index(inplace=True, drop = True)
        terms_for_node2vec['index_mapping'] = terms_for_node2vec.index
    else:
        terms_for_node2vec = valid_go_terms[["id", "is_a" ]]
        terms_for_node2vec['id'] = terms_for_node2vec['id'].apply(lambda x: x.strip("['']")) 
        terms_for_node2vec['is_a'] = terms_for_node2vec['is_a'].apply(lambda x:  x.strip("[']").replace(' ', '').split("','") if type(x) is str else x) 

        terms_for_node2vec.reset_index(inplace=True, drop = True)
        terms_for_node2vec['index_mapping'] = terms_for_node2vec.index
    
    return terms_for_node2vec


def create_edge_list(terms_for_node2vec, onto_type='go'):
    
    """
    Function that takes all the node2vec terms
    adds all the relationships of type 'is_a' and 'part_of'
    :return lists of all the edges
    """
    
    is_a_dict = dict(zip(terms_for_node2vec["index_mapping"].values,
                     terms_for_node2vec["is_a"].values))
    if onto_type=='go':
        part_of_dict = dict(zip(terms_for_node2vec["index_mapping"].values,
                     terms_for_node2vec["relationship"].values))
    go_to_index_dict = dict(zip(terms_for_node2vec["id"].values,
                     terms_for_node2vec["index_mapping"].values))

    go_graph_edges = defaultdict(list)

    #adding all the 'is_a' edges
    for i, is_a_list in is_a_dict.items():
        if type(is_a_list) is list: #non root GO term that does not have a 'is_a'
            for is_a in is_a_list:
                if type(is_a) is str:
                    go_graph_edges[i].append(go_to_index_dict[is_a])            
    if onto_type=='go':
        #adding all the 'part_of' edges
        for i, part_of_list in part_of_dict.items():
            if type(part_of_list) is list: # no relationship present
                for part_of in part_of_list:
                    if type(part_of) is str and "part_of" in part_of:
                        part_of =  part_of.strip("'part_of ").replace("''", "")
                        go_graph_edges[i].append(go_to_index_dict[part_of])    
    return go_graph_edges


def write_edge_list(go_graph_edges, save_path = "graph/go-terms.edgelist"):
    
    """Writes all the GO 'is_a' and 'part_of' to a file: ex: 1->2
    Args:
        go_graph_edges (dict): dict of GO relations ex. GO -> [GO1, GO2, ...] 
    """
    
    with open(save_path, "w") as f:  
        for node, edge_list in go_graph_edges.items():
            for edge in edge_list:
                #adding 1 as the weight
                f.write(str(node) + "  " + str(edge)) #+ " " + str(1)) 
                f.write("\n")
    return


def save_go_mapping(terms_for_node2vec, save_path = 'go_id_dict'):
    go_to_index_dict = dict(zip(terms_for_node2vec["id"].values,
                     terms_for_node2vec["index_mapping"].values))
    with open(save_path, 'wb') as fp:
        pickle.dump(go_to_index_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return


folder_path = '/home/Data/'
filename = folder_path + 'go.obo' # hp.obo # doid.obo

obo_dict, unique_tags = obo_file_to_dict(filename)
obo_panda = obo_dict_to_pandas(obo_dict, unique_tags)   
obo_panda.to_csv(filename + ".csv", index=False)


terms_for_node2vec = obo_csv_trim(csv_path =filename + '.csv', onto_type='go')  # ["id", "is_a"], 
save_go_mapping(terms_for_node2vec, save_path = folder_path + 'go_id_dict')
# goid: index
go_to_index_dict = dict(zip(terms_for_node2vec["id"].values, terms_for_node2vec["index_mapping"].values)) 
 
go_graph_edges = create_edge_list(terms_for_node2vec, onto_type='go')
write_edge_list(go_graph_edges, save_path = folder_path + "go-terms.edgelist")

 