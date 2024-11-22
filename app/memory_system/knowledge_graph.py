import igraph

class KnowledgeGraph:
    def __init__(self):
        self.graph = igraph.Graph(directed=True)

    def add_connections_via_feedback(self, embeddings, labels):
        from sklearn.semi_supervised import LabelSpreading

        label_spreading = LabelSpreading(kernel="knn", alpha=0.8)
        label_spreading.fit(embeddings, labels)
        predicted_labels = label_spreading.transduction_

        for i, label in enumerate(predicted_labels):
            if label >= 0:
                source = self.graph.vs[i]["name"]
                target = self.graph.vs[label]["name"]
                self.add_edge(source, target, weight=0.5)

    def add_edge(self, source_id, target_id, weight=1.0):
        if not self.graph.are_connected(source_id, target_id):
            self.graph.add_edge(source_id, target_id, weight=weight)