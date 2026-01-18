from graph_builder import KnowledgeGraphBuilder
from config import LLM_MODEL_PATH

if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    builder.run_update()
    builder.close()

#     from langchain_community.llms import LlamaCpp
#     from langchain_core.prompts import PromptTemplate
#     from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
#     from pydantic.v1 import BaseModel, Field

#     llm = LlamaCpp(
#         model_path=str(LLM_MODEL_PATH),
#         n_batch=512,
#         temperature=0.0,
#         n_gpu_layers=-1,
#         n_ctx=8192,
#         verbose=False,
#         backend="vulkan",
#         max_tokens=1000,
#     )

#     class Relationship(BaseModel):
#         reasoning: str = Field(
#             description="A brief explanation for the chosen relationship."
#         )
#         relation_type: str = Field(
#             description="The single most appropriate relationship type."
#         )

#     parser = JsonOutputParser(pydantic_object=Relationship)

#     prompt = PromptTemplate(
#         template="""
# You are a helpful assistant that provides structured data.
# Analyze the user's query and generate a JSON object based on the following instructions.
# Your entire response must be a single, valid JSON object and nothing else. Do not include any conversational text, preamble, or explanations.

# {format_instructions}

# Here is the user's query:
# {query}
#     """,
#         input_variables=["query"],
#         partial_variables={"format_instructions": parser.get_format_instructions()},
#     )

#     chain = prompt | llm | parser

#     try:
#         response = chain.invoke(
#             {"query": "Analyze the relationship between a student and a teacher."}
#         )
#         res = Relationship.validate(response)
#         print(res.relation_type)

#     except Exception as e:
#         print("Parsing failed. The model's raw output was:")
#         print(e.llm_output)

#     llm.client.close()
#     del llm

#     import gc

#     gc.collect()
