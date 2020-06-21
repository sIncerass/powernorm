import sys
import json
import os
#node_to_rank = []
"""
with open("host_file.txt", "w") as out:
    for i, line in enumerate(sys.stdin):
        if i == 0:
            os.environ['MASTER_HOST'] = line
        else:
            out.write( "%s" % line)
    #node_to_rank.append(i)
#json.dump(node_to_rank, open('node_to_rank.json', 'w'))
"""
node_to_rank = {}
with open("host_file.txt", "w") as out:
    for i, line in enumerate(sys.stdin):
        node_to_rank[line[:-1]] = i
        out.write( "%s" % line)
json.dump(node_to_rank, open('node_to_rank.json', 'w'))
