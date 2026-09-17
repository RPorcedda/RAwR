import sys
import subprocess
import numpy as np

# PARAM
# filename: network dataset name (it is expected a directory format like dataset/dataset.edgelist)
# num_nodes: number of nodes in network
# eps : tolerance from 0 (exact EP) to max degree (only one block. The same partition could be obtained also for a value less than max degree)

#OUTPUT
# num_blocks: number of blocks in partition
# block_sizes: number of nodes in each block
# part: array of length num_nodes which specifies to what block in the partition each node belongs to
# partition: list with num_blocks rows, each row contains nodes indices (starting from 0) of nodes in the block
def computeBEpartition(filename, num_nodes, eps):
    print(f"eps-BE started on {filename} with epsilon={eps}")
    
    # set the parameters for one-shot eps-BE (not the iterative version)
    eps_0=eps
    delta=eps
    delta_max=eps
    if eps==0:
        delta=1

    result = subprocess.run(
        ["java", "-jar", "epsBE.jar",
            filename+"/"+filename+".edgelist", 
            str(num_nodes), 
            str(eps_0),
            str(delta_max),
            str(delta),
            "partitions/"+filename+"BE"+str(eps),
            "false",
            "false"],
        capture_output=True,
        text=True,
        check=True
    )
        
    #print(result.stdout)
    #print(result.stderr)
    f = open("partitions/"+filename+"BE"+str(eps),"r")

    line=f.readline()
    line=f.readline()
    f.close()
    line = line.split(",")
    part = np.zeros(num_nodes, dtype=int)
    block_sizes = []

    blockIndex = 0
    partition = []
    for elem in line:
        block = []
        elem = elem.split(" ")
        elem = elem[1:len(elem)-1]
        tot = 0
        inn = False
        for el in elem:
            inde = int(el.replace("x",""))
            part[inde-1] = blockIndex
            block.append(inde-1)
            tot = tot+1
            inn = True
        if(inn):
            partition.append(block)
            block_sizes.append(tot)
        blockIndex = blockIndex+1
    num_blocks = len(block_sizes)
    block_sizes = np.array(block_sizes)

    print("eps-BE concluded")
    print(f"Number of blocks in partition: {num_blocks}")
    print(f"Block sizes:\n{block_sizes}")
    with open(f"partitions/{filename}P{eps}", "w") as f:
        for line in part:
            f.write("%s\n" % line)
        f.close()

    return num_blocks, block_sizes, part, partition



import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=True, type=str)
    parser.add_argument("--num_nodes", required=True, type=int)
    parser.add_argument("--eps", required=True, type=int)
    args = parser.parse_args()

    num_blocks, block_sizes, part, partition = computeBEpartition(args.filename, args.num_nodes, args.eps)