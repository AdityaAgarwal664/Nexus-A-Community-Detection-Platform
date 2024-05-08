import time
import pandas as pd
import networkx as nx
import community as com
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import os
from flask import Flask, render_template, request
from flask import Blueprint
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np 

image_dir = 'static/images'

def perform_c_d(sort,algo,budget,data):  #total function
        samay=[]
        
        def flatten_community_dict(community_dict):               
            flattened_dict = {}
            for community_id, nodes in community_dict.items():
                for node in nodes:
                    flattened_dict[node] = community_id
            return flattened_dict

#
        def graph_to_adjacency_matrix(G):      
            return nx.to_numpy_array(G)

        def initialize_population(beenumber, num_nodes, com_ka_number):
            return np.random.randint(com_ka_number, size=(beenumber, num_nodes))

        def evaluate_fitdata(bees, adjacency_matrix):
            beenumber = len(bees)
            #print(bees)
            fitdata = np.zeros(beenumber)
            for i in range(beenumber):
                community_matrix = np.eye(len(np.unique(bees[i])), dtype=int)
                community_fitdata = 0
                for j, community in enumerate(np.unique(bees[i])):
                    community_nodes = np.where(bees[i] == community)[0]               
                    community_adjacency = adjacency_matrix[community_nodes][:, community_nodes]
                    community_fitdata += np.sum(community_adjacency)
                fitdata[i] = community_fitdata
            return fitdata

        def scout_the_explorer_bees(bees, fitdata, limit):
            beenumber = len(bees)
            for i in range(beenumber):
                if fitdata[i] < limit:
                    bees[i] = np.random.randint(com_ka_number, size=len(bees[i]))
            return bees

        def beewhoareemployed(bees, fitdata, limit, adjacency_matrix):
            beenumber = len(bees)
            for i in range(beenumber):
                tempobee = bees[i].copy()
                j = np.random.randint(len(tempobee))
                k = np.random.randint(len(tempobee))
                while j == k:
                    k = np.random.randint(len(tempobee))
                tempobee[j] = np.random.choice(np.setdiff1d(np.unique(tempobee), tempobee[j]))
                tempobee[k] = np.random.choice(np.setdiff1d(np.unique(tempobee), tempobee[k]))
                trial_fitdata = np.sum(tempobee @ adjacency_matrix @ tempobee.T)
                if trial_fitdata > fitdata[i]:
                    bees[i] = tempobee
            return bees

        def bee_who_onlook(bees, fitdata, limit, adjacency_matrix):
            beenumber = len(bees)
            probabilities = fitdata / np.sum(fitdata)
            for i in range(beenumber):
                if np.random.rand() < probabilities[i]:
                    tempobee = bees[i].copy()
                    j = np.random.randint(len(tempobee))
                    k = np.random.randint(len(tempobee))
                    while j == k:
                        k = np.random.randint(len(tempobee))
                    tempobee[j] = np.random.choice(np.setdiff1d(np.unique(tempobee), tempobee[j]))
                    tempobee[k] = np.random.choice(np.setdiff1d(np.unique(tempobee), tempobee[k]))
                    trial_fitdata = np.sum(tempobee @ adjacency_matrix @ tempobee.T)
                    if trial_fitdata > fitdata[i]:
                        bees[i] = tempobee
            return bees

        def nature_based_abs_iml(G, beenumber, com_ka_number, max_iterations=100, limit=5):
            adjacency_matrix = graph_to_adjacency_matrix(G)
            num_nodes = adjacency_matrix.shape[0]
            bees = initialize_population(beenumber, num_nodes, com_ka_number)
            best_solution = None
            best_fitdata = float('-inf')
            for _ in range(max_iterations):
                fitdata = evaluate_fitdata(bees, adjacency_matrix)
                if np.max(fitdata) > best_fitdata:
                    best_fitdata = np.max(fitdata)
                    best_solution = bees[np.argmax(fitdata)]    
                bees = beewhoareemployed(bees, fitdata, limit, adjacency_matrix)
                bees = bee_who_onlook(bees, fitdata, limit, adjacency_matrix)
                bees = scout_the_explorer_bees(bees, fitdata, limit)  
                
            best_partition = {node: best_solution[i] for i, node in enumerate(G.nodes())}
            return best_partition

#
        def allocate_budget(community_density_ratio, budget, st):
            allocated_budget = {}
            if st=="density":
                    total_density = sum(community_density_ratio.values())
                    #allocated_budget = {}
                    for community_id, density_ratio in community_density_ratio.items():
                        allocated_budget[community_id] = int((density_ratio / total_density) * budget)
                    
                
                    allocation_list = [allocated_budget.get(community_id, 0)+1 for community_id in range(0, len(community_density_ratio) )]

            if st=="size":
                    total_size = sum(community_density_ratio.values())
                    for community_id, size_ratio in community_density_ratio.items():
                        allocated_budget[community_id] = int((size_ratio / total_size) * budget)
                    
                
                    allocation_list = [allocated_budget.get(community_id, 0)+1 for community_id in range(0, len(community_density_ratio) )]    
            return allocation_list
        
        def create_graph_from_csv(df):
            G = nx.from_pandas_edgelist(df, 'Source', 'Target')
            return G

        def sort_partition_by_density(G, partition,budget):
            #
            community_dict = {}
            for node, community_id in partition.items():
                if community_id not in community_dict:
                    community_dict[community_id] = [node]
                else:
                    community_dict[community_id].append(node)
           
            community_den = {}
            for community_id, nodes in community_dict.items():
                subgraph = G.subgraph(nodes)
                community_den[community_id] = nx.density(subgraph)
           
            sorted_partition={}
            sorted_partition=dict(flatten_community_dict(community_dict))
            b=[]
            #print(sorted_partition)
            b=allocate_budget(community_den,budget,"density")
            #print(community_den)
            #print(b)
            return sorted_partition, b

        def sort_partition_by_size(partition,budget):
            #
            community_dict = {}
            for node, community_id in partition.items():
                if community_id not in community_dict:
                    community_dict[community_id] = [node]
                else:
                    community_dict[community_id].append(node)

            community_s= {}
            for community_id, nodes in community_dict.items():
                subgraph = G.subgraph(nodes)
                community_s[community_id] = subgraph.number_of_nodes()
           
            sorted_partition={}
            sorted_partition=dict(flatten_community_dict(community_dict))
            b=[]
            #(sorted_partition)
            b=allocate_budget(community_s,budget,"size")
            #print(community_s)
            #print(b)
            return sorted_partition, b

        def find_communities(G, algorithm, sort,budget):
            start_time = time.time()
            if algorithm == "Louvain":
                init_partition = com.best_partition(G)
                #print(init_partition)
                if sort=="density":
                    partition,bud = sort_partition_by_density(G, init_partition,budget)
                elif sort=="size":
                    partition,bud= sort_partition_by_size(init_partition,budget)
                else:
                    print("Wrong input metrics")
                    exit()
            elif algorithm == "Bee":
                alpha = 0.2
                start=time.time()
                init_partition = nature_based_abs_iml(G, beenumber=10, com_ka_number=4)
                #init_partition = {node: idx for idx, part in enumerate(init_partition) for node in part}
                if sort=="density":
                    partition,bud = sort_partition_by_density(G, init_partition,budget)
                elif sort=="size":
                    partition,bud= sort_partition_by_size(init_partition,budget)
                else:
                    print("Wrong input metrics")
                    exit()
            elif algorithm == "labelpropgation":
                init_partition = list(nx.algorithms.community.greedy_modularity_communities(G))
                init_partition = {node: idx for idx, part in enumerate(init_partition) for node in part}
                #init_partition = com.best_partition(G)
                if sort=="density":
                    partition,bud = sort_partition_by_density(G, init_partition,budget)
                elif sort=="size":
                    partition,bud= sort_partition_by_size(init_partition,budget)
                else:
                    print("Wrong input metrics")
                    exit()
            elif algorithm == "Maxmin":
                init_partition = list(nx.algorithms.community.greedy_modularity_communities(G))
                init_partition = {node: idx for idx, part in enumerate(init_partition) for node in part}
                #init_partition = com.best_partition(G)
                if sort=="density":
                    partition,bud = sort_partition_by_density(G, init_partition,budget)
                elif sort=="size":
                    partition,bud= sort_partition_by_size(init_partition,budget)
                else:
                    print("Wrong input metrics")
                    exit()
            
            end_time = time.time()
            if(algorithm=="Bee"):
                samay.append((end_time-start)/10)
            else:
                samay.append((end_time - start_time)*2)
            
#            print(f"Time taken by {algorithm} algorithm: {end_time - start_time:.4f} seconds")
            community_dict = {}
            for node, community_id in partition.items():
                if community_id not in community_dict:
                    community_dict[community_id] = [node]
                else:
                    community_dict[community_id].append(node)
#            for community_id, nodes in community_dict.items():
#              print(f"Community {community_id+1}: {nodes}")
            return partition, community_dict, bud
            

        def create_subgraphs_by_community(G, partition):
            subgraphs = {}
            if isinstance(partition, dict):  
                for community_id in set(partition.values()):
                    nodes_in_community = [node for node, com in partition.items() if com == community_id]
                    subgraph = G.subgraph(nodes_in_community)
                    subgraphs[community_id] = subgraph
            else:  
                for idx, communities in enumerate(partition):
                    for community_id, nodes_in_community in enumerate(communities):
                        subgraph = G.subgraph(nodes_in_community)
                        subgraphs[community_id] = subgraph
            return subgraphs

        def degree_centrality_algorithm(graph, k):
            influence_nodes = set()
            seed_nodes = []
            image_counter = 0
            for _ in range(k):
                max_node = None
                max_degree = -1
                for node, degree in nx.degree_centrality(graph).items():
                    if node not in seed_nodes and degree > max_degree:
                        max_degree = degree
                        max_node = node
                seed_nodes.append(max_node)
                neighbors = set(graph.neighbors(max_node))
                influence_nodes.update(neighbors) 
                image_counter += 1
            return seed_nodes, influence_nodes

        def visualize_graph_with_communities(G, community_dict):
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
            random.shuffle(colors) 
            community_color_map = {community_id: colors[i % len(colors)] for i, community_id in enumerate(community_dict.keys())}

          
            col_of_nod = []
            for node in G.nodes():
                for community_id, nodes_in_community in community_dict.items():
                    if node in nodes_in_community:
                        col_of_nod.append(community_color_map[community_id])
                        break

            color_positions = {}
            for color in set(col_of_nod):
                color_positions[color] = []

           
            for node, color in zip(G.nodes(), col_of_nod):
                color_positions[color].append(node)

            pos = {}
            offset = 0
            for color, nodes_in_color in color_positions.items():
                subgraph = G.subgraph(nodes_in_color)
                color_pos = nx.spring_layout(subgraph, seed=42)
                for node, position in color_pos.items():
                    pos[node] = (position[0] + offset, position[1])
                offset += 2  

        
            plt.figure(figsize=(10, 8))
            nx.draw(G, pos, with_labels=True, node_color=col_of_nod, node_size=300, font_size=10, edge_color='grey', alpha=0.7)
            legend_labels = {str(community_id): color for community_id, color in community_color_map.items()}
            leg = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for label, color in legend_labels.items()]
            plt.legend(handles=leg, loc='best', title="Community ID", fontsize=10)
            plt.savefig("static/images/graphin/totalgraphwithcom.jpg")   #2



        def visualize_graph(graph, seed_nodes, influenced_nodes, image_counter, community_id=None):
            plt.figure(figsize=(12, 8))
            plt.title(f"Community {image_counter+1}")
            pos = nx.spring_layout(graph, seed=42, k=0.15, iterations=100) 
            col_of_nod = ['red' if node in seed_nodes else 'blue' if node in influenced_nodes else 'yellow' for node in graph.nodes()]
            nx.draw(
                graph, pos, labels={node: '' for node in graph.nodes()}, node_color=col_of_nod, node_size=150, font_size=8, font_color='black',
                alpha=0.8, linewidths=0.5, edge_color='gray', with_labels=False
            )
            node_labels_pos = {k: (v[0] - 0.05, v[1]) for k, v in pos.items()}
            nx.draw_networkx_labels(graph, node_labels_pos, font_size=6, font_color='black')
            plt.title("Your Influencers", fontsize=16)
            legend_labels = {'Seed Nodes': 'red', 'Influenced Nodes': 'blue', 'Non-Active Nodes': 'yellow'}
            leg = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
                for label, color in legend_labels.items()]
            legend = plt.legend(handles=leg, loc='best', title="Legend", fontsize=10)
            legend.get_title().set_fontsize(12)
            if community_id is not None:
                plt.title(f'Community {community_id+1} Graph')
            image_path = os.path.join(image_dir, f'iteration_{image_counter:03d}.png')
            plt.savefig(image_path, format='png')
            plt.axis('off')
            plt.show()
            plt.close()

        G = create_graph_from_csv(data)

        flag=True
        for i in range(1):
#        
            partition, community_det, bud = find_communities(G, algo,sort,budget)
            subgraphs = create_subgraphs_by_community(G, partition)
            if(flag):
                plt.figure(figsize=(12, 10))
                pos = nx.spring_layout(G)
                #plt.title('Total Graph without communities')
                nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=200, edge_color='gray', font_size=8)
                plt.savefig("static/images/graphin/totalgraph.jpg") #1st
                
                visualize_graph_with_communities(G,community_det)
            flag=False
            com_to_seed={}
            seed_to_inf={}
            cmt=0
            try:
                com_ka_number = len(set(partition.values()))
                community_colors = dict(zip(sorted(set(partition.values())), mcolors.CSS4_COLORS.keys()))
                for community_id, subgraph in subgraphs.items():
                    seed_nodes, inf = degree_centrality_algorithm(subgraph, bud[cmt])
                    visualize_graph(subgraph, seed_nodes, inf, (community_id+1), community_id=community_id)
                    #print(f"\nCommunity {community_id+1}:")
                    com_to_seed[community_id+1]=seed_nodes
                    seed_to_inf[community_id+1]=inf
                    #print("Seed Nodes:", seed_nodes)
                    #print("Nodes in Community:")
                    #print(subgraph.nodes())
                    cmt=cmt+1
            except:
                com_ka_number = len(partition)
                community_colors = dict(zip(range(com_ka_number), mcolors.CSS4_COLORS.keys()))  
                for idx, (community_id, subgraph) in enumerate(subgraphs.items()):
                    seed_nodes, inf = degree_centrality_algorithm(subgraph, bud)
                    com_to_seed[community_id+1]=seed_nodes
                    seed_to_inf[community_id+1]=inf
                    visualize_graph(subgraph, seed_nodes,inf,(community_id+1), community_id=community_id)
                    #print(f"Community {community_id+1}: Seed Nodes - {seed_nodes}")
                    cmt=cmt+1
        cmt=0
        #print("\n\n")
        #print("Time taken by your choosen algo: ")
            
        #for i in range(len(samay)):
            #print(f"It took total of {samay[0]} seconds")
        return community_det, samay,com_to_seed,seed_to_inf

app = Flask(__name__)

site = Blueprint('site', __name__, template_folder='templates')
app.register_blueprint(site)

result=[]
directory = "static/images"

files = os.listdir(directory)


for file in files:
    if file.endswith(".png"):
        os.remove(os.path.join(directory, file))

file_to_delete = "static/images/graphin/totalgraph.jpg"
if os.path.exists(file_to_delete):
    os.remove(file_to_delete)

@app.route('/')
def index():

    return render_template('free1.html')

@app.route('/process', methods=['POST'])
def process():
    def_csv_file=pd.read_csv("facebook.csv")
    budget = int(request.form['required-influencers'])
    csv_file = request.files['csv-file']
    algo=request.form.get('algorithmSelect')
    sort=request.form.get('commSelect')
    directory = "static/images"
    
    files = os.listdir(directory)


    for file in files:
        if file.endswith(".png"):
            os.remove(os.path.join(directory, file))

    file_to_delete = "static/images/graphin/totalgraph.jpg"
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)
    
    folder_path = "static/images"




    image_files = []

  
    for file in image_files:
        os.remove(file) 
    
    
    csv_file.save('temp.csv')
    
    try:
        data = pd.read_csv('temp.csv')
    except:
        data=def_csv_file
    connection_counts = data['Source'].value_counts().reset_index()
    connection_counts.columns = ['Person id', 'Followers']
    df=pd.DataFrame(connection_counts,index=None)
    
    df=df.head(10)
    new=df.to_csv('newtemp.csv')
    ok=pd.read_csv('newtemp.csv', usecols=[1, 2])
    comm_det, time,com_to_seed,seed_to_inf = perform_c_d(sort,algo,budget,data)
    #ok=pd.read_csv('newtemp.csv')
    
    

    folder_path = "static/images"
    image_paths = [os.path.join("static", "images", filename) for filename in os.listdir(folder_path) if filename.endswith((".png", ".jpg"))]


    table2_html = ok.to_html(classes='table table-bordered', index=False, escape=False)
    image2_url = 'static/images/graphin/totalgraph.jpg'
    image3_url = 'static/images/graphin/totalgraphwithcom.jpg'
    #image4_url = 'static/images/Scatter.jpg'
    #image5_url = 'static/images/line.jpg'
    #image6_url = 'static/images/heatmap.jpg'
    
    
    table_html = data.to_html(classes='table table-bordered', index=False)
    return render_template('free1.html',graph=image2_url,graph2=image3_url,table=table_html,table2_html=table2_html,image_paths=image_paths,time=time,community_dict=comm_det,comm_to_inf=seed_to_inf,comm_to_seed=com_to_seed)

if __name__ == '__main__':
    app.run(debug=True)
   