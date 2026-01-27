import os
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed


# @st.cache_resource
def get_lightrag_engine(working_dir, api_key, embedding_model, embedding_dim):
    """
    Initializes the LightRAG engine.
    """
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: gpt_4o_mini_complete(
            prompt, 
            system_prompt=system_prompt, 
            history_messages=history_messages, 
            api_key=api_key, 
            **kwargs
        ),        
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim, 
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts, 
                model=embedding_model, 
                api_key=api_key        
            ) 
        ),        
        llm_model_name="gpt-4o-mini",
    )    
    return rag