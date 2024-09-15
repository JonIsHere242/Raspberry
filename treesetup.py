import openai
import json
import os
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import random
import networkx as nx
import matplotlib.pyplot as plt
import re

"""
    This file implements a Tree of Thought system using Monte Carlo Tree Search (MCTS).
    It generates and evaluates possible solutions to a given problem using language models.

    ASCII representation of MCTS with scores (1-100):

                  Root (50)
                /          \
          Node A (30)    Node B (70)
            /         \          \
    Node C (20)   Node D (40)    Node E (80)

    During backpropagation, scores propagate upwards:
    Node E (80) -> Node B (70) -> Root (50)

    The system uses language models to generate and evaluate solutions,
    visualizing the resultant tree of thought.

"""





class Node:
    def __init__(self, content: str, parent=None):
        self.content = content
        self.parent = parent
        self.children: List['Node'] = []
        self.evaluation: Dict[str, float] = {}
        self.visits = 0
        self.value = 0


class TreeOfThought:
    def __init__(self, api_key: str, base_url="http://localhost:3301/v1", generator_model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", evaluator_model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.generator_model = generator_model
        self.evaluator_model = evaluator_model
        self.graph = nx.DiGraph()  # Create a directed graph for visualization

    def generate_children(self, node: Node, num_children: int = 2) -> List[Node]:
        prompt = f"Given the following problem and partial solution, provide {num_children} possible next steps or continuations:\n\nProblem: {node.problem}\n\nPartial solution: {node.content}"

        response = self.client.chat.completions.create(
            model=self.generator_model,
            messages=[{"role": "user", "content": prompt}],
            n=num_children
        )

        # Access the response content correctly
        children = [Node(choice.message.content, parent=node) for choice in response.choices]
        for child in children:
            child.problem = node.problem
        node.children = children
        return children



    def evaluate_node(self, node: Node) -> Dict[str, float]:
        prompt = f"""
        Evaluate the following solution based on three criteria: correctness, relevance, complexity.
        Each score should be between 0 and 100. Reply ONLY with three numbers separated by spaces.
        Example: '15 58 98'
        Example: '76 32 45'
        Example: '89 35 43'
        Example: '24 43 56'
        Example: '90 70 70'
        Do not include any other text in your response.

        Problem: {node.problem}

        Solution: {node.content}
        """

        response = self.client.chat.completions.create(
            model=self.evaluator_model,
            messages=[{"role": "user", "content": prompt}]
        )

        # Try to extract and validate the response
        evaluation_text = response.choices[0].message.content.strip()

        # Use regex to find the first occurrence of three numbers separated by spaces
        match = re.search(r'(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})', evaluation_text)

        if not match:
            raise ValueError(f"Could not find valid scores in the response: {evaluation_text}")

        # Extract the scores from the match and convert them to float
        scores = list(map(float, match.groups()))

        if not all(self.is_valid_score(score) for score in scores):
            raise ValueError(f"Scores out of valid range (0-100): {scores}")

        node.evaluation = {
            "correctness": scores[0],
            "relevance": scores[1],
            "complexity": scores[2]
        }
        print(f"Scores for node: {node.evaluation}")
        return node.evaluation

    # Helper function to check if the score is a valid float between 0 and 100
    def is_valid_score(self, score: float) -> bool:
        return 0 <= score <= 100



    def uct_select(self, node: Node) -> Node:
        if not all(child.visits > 0 for child in node.children):
            return random.choice([child for child in node.children if child.visits == 0])

        log_total_visits = np.log(sum(child.visits for child in node.children))
        uct_values = [
            (child.value / child.visits) + np.sqrt(2 * log_total_visits / child.visits)
            for child in node.children
        ]
        return node.children[np.argmax(uct_values)]

    def mcts(self, root: Node, num_simulations: int = 20, max_depth: int = 10):
        for _ in tqdm(range(num_simulations), desc="MCTS Simulations"):
            node = root
            path = []

            # Selection
            while node.children and node.visits > 0:
                node = self.uct_select(node)
                path.append(node)

            # Expansion
            if not node.children and len(path) < max_depth:
                self.generate_children(node)
                if node.children:
                    node = random.choice(node.children)
                    path.append(node)

            # Simulation
            if not node.evaluation:
                self.evaluate_node(node)

            # Backpropagation
            value = sum(node.evaluation.values())
            for n in path:
                n.visits += 1
                n.value += value

        return self.get_best_path(root)

    def get_best_path(self, root: Node) -> List[Node]:
        path = [root]
        current = root
        while current.children:
            current = max(current.children, key=lambda c: c.value / c.visits if c.visits > 0 else 0)
            path.append(current)
        return path

    def solve_problem(self, problem: str) -> (List[str], Node):
        root = Node(problem)
        root.problem = problem
        best_path = self.mcts(root)
        return [node.content for node in best_path], root

    
    def visualize_tree(self, root: Node):
        self.graph.clear()
        self._build_graph(root)

        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(10, 10))

        node_colors = [self.graph.nodes[node]["visits"] for node in self.graph.nodes()]
        cmap = plt.cm.Blues

        nodes = nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            cmap=cmap,
            node_size=700,
            vmin=min(node_colors),
            vmax=max(node_colors)
        )
        edges = nx.draw_networkx_edges(self.graph, pos)

        labels = {
            node: f"Correctness: {int(self.graph.nodes[node]['correctness'])}%\n"
                  f"Relevance: {int(self.graph.nodes[node]['relevance'])}%\n"
                  f"Complexity: {int(self.graph.nodes[node]['complexity'])}%"
            for node in self.graph.nodes()
        }
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)

        plt.colorbar(nodes, label="Visits")
        plt.title("Tree of Thought Visualization")
        plt.axis('off')
        plt.show()




    def _build_graph(self, node: Node):
        self.graph.add_node(node, visits=node.visits, correctness=node.evaluation.get("correctness", 0),
                            relevance=node.evaluation.get("relevance", 0), complexity=node.evaluation.get("complexity", 0))
        for child in node.children:
            self.graph.add_edge(node, child)
            self._build_graph(child)

def load_problem(problem_number: int) -> str:
    folder_path = f"problem_sets/problem_{problem_number}"
    json_file_path = os.path.join(folder_path, "problem_data.json")
    
    with open(json_file_path, 'r') as json_file:
        problem_data = json.load(json_file)
    
    return problem_data["problem_statement"]




# Usage
api_key = "lm-studio"  # Local LLM Studio API Key

##only use a local llm unless you want to spend a lot of money
##tree of thought will use like 100k tokens per problem hopefully we can use the final result to make a good 0 shot learner
##I believe this is how the Q* synthetic data was generated and the chain of thought model was trained to aproximate the Q* results
tot_system = TreeOfThought(api_key)
problem = load_problem(0)  # Load problem 0
solution, root_node = tot_system.solve_problem(problem)

# Visualize the tree
tot_system.visualize_tree(root_node)

