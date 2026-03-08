from .graph_builder import KnowledgeGraphBuilder

if __name__ == "__main__":
    builder = KnowledgeGraphBuilder(fresh_start=True, use_api=False)
    builder.run_update()
    builder.close()
