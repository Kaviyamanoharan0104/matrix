import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st
from openai import OpenAI
from typing import Optional
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.core.memory import ChatMemoryBuffer
import json
from cryptography.fernet import Fernet
import plotly.express as px
import plotly.graph_objects as go
from vanna.chromadb import ChromaDB_VectorStore
from vanna.openai import OpenAI_Chat
import asyncio
 
 
class AthenaAgent(Workflow):
    def __init__(self,vn, timeout: Optional[float] = 200.0):
        super().__init__(timeout=timeout)
        self.memory_VM_OPs= ChatMemoryBuffer.from_defaults(token_limit=3900)
        with open('secret.key', 'rb') as key_file:
            key = key_file.read()
 
        cipher_suite = Fernet(key)
 
        # Load the encrypted configuration data
        with open('config.json', 'r') as config_file:
            encrypted_data = json.load(config_file)
 
        # Decrypt the sensitive information
        data = {key: cipher_suite.decrypt(value.encode()).decode() for key, value in encrypted_data.items()}
       
        self.api_key = data["API_KEY"]
       
        # Athena configuration
        self.aws_access_key = data["aws_access_key"]
        self.aws_secret_key = data["aws_secret_key"]
        self.region_name = "eu-north-1"
        self.athena_db_name = "matrixdb"
        self.s3_output_location = "s3://matrixresult/"
       
        # Athena connection
        self.athena_engine = create_engine(
            f'awsathena+rest://{self.aws_access_key}:{self.aws_secret_key}@athena.{self.region_name}.amazonaws.com/{self.athena_db_name}?s3_staging_dir={self.s3_output_location}'
        )
 
        # vn initialization
        self.vn = vn
        self.vn.run_sql = self.run_query
        self.vn.run_sql_is_set = True
 
    def run_query(self, sql: str) -> pd.DataFrame:
        """Execute a query against Athena and return the result as a DataFrame."""
       
        with self.athena_engine.connect() as connection:
            result = connection.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df
       
           
 
    def get_schema(self) -> dict:
        """Retrieve schema information for documentation and querying."""
       
        schema_query = """
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME IN ('outputfilevm', 'outputfilevmdisk')
        """
        df_schema = self.run_query(schema_query)
       
        schema = {}
        for table_name, group in df_schema.groupby('TABLE_NAME'):
            schema_details = group.to_dict(orient='records')
 
            # Fetch sample data
            sample_query = f"SELECT * FROM {table_name} LIMIT 5"
            try:
                sample_data = self.run_query(sample_query).to_dict(orient='records')
            except Exception as e:
                sample_data = f"Error fetching sample data: {e}"
 
            schema[table_name] = {
                "columns": schema_details,
                "sample_data": sample_data
            }
        return schema
       
 
 
    def engineer_prompt(self, question: str, schema: dict) -> str:
        """Generate a prompt for SQL query generation based on the schema and user question."""
        return f"""
        You are a SQL expert tasked with generating a query based on the provided schema. It is critical to adhere strictly to the schema and use the exact table and column names as specified.
 
**Database Name**: {self.athena_db_name}
 
**Schema**:
{schema}
 
**Task**:
1. **Objective**: Write an SQL query to address the following question:
   *"{question}"*
 
2. **Pre-query Requirements**:
   - **Read-only operation**: The query must only perform read operations. No modifications such as `INSERT`, `UPDATE`, or `DELETE` are permitted.
    - **Schema Validation**: Carefully review the schema to:
        - Identify the correct tables and columns relevant to the question.
        - Verify all table and column names match exactly as specified in the schema.
        - Understand the relationships between tables (if applicable).
        - **Always use `GROUP BY` for all the queries** (if aggregation is required).
        - Ensure that **all column names are fully qualified** using the format `table_name.column_name` to avoid ambiguity.
 
3. **Query Construction Guidelines**:
   - All the queries are asked in the context of Nexturn,therefore omit the word "Nexturn"
    - Use only the table names, column names, and relationships explicitly provided in the schema.
    - Fully qualify all column names with their respective table names, especially when columns with the same name appear in multiple tables.
    - If the schema does not contain all the necessary information for the query, provide a detailed explanation of why the query cannot be completed.
    - Use `AS` to alias columns or tables when appropriate for clarity, but do not deviate from the schema.
    - Convert the following natural language query into an SQL query. When filtering text columns, use the CONTAINS clause instead of the WHERE clause with = or LIKE, unless an exact match is explicitly requested. Ensure the query is correctly formatted for execution.
    - Use the `FROM database_name.table_name` format to qualify table names.
    - For filtering text columns based on user queries:
      - **Prefer `LIKE '%keyword%'`** when searching for words in text columns.
      - If available, **use `CONTAINS(column_name, 'keyword')`** for full-text search.
      - **Do NOT use `=` for text-based searches** unless checking for an exact match.
    - Use `DISTINCT` to ensure unique values where necessary.
    -Example:list down all the FP contracts at Nexturn-"SELECT * FROM your_tableWHERE your_column LIKE '%FP%"
 
 
4. **Validation Checklist**:
   - Double-check that all table and column names in the query exactly match the schema.
    - Ensure that relationships between tables (if used) align with those described in the schema.
    - Confirm that the query fully addresses the question within the constraints of the schema.
    - Confirm that all column references are fully qualified to prevent ambiguity errors.
    - Verify that text searches use CONTAINS() for better performance with long text fields.
 
5. **Output**:
   - If the query can be written, return only the SQL query.
   - If the query cannot be generated due to insufficient or unclear schema information, provide a detailed rationale for why it is not possible.
 
**Note**: Adherence to the schema is mandatory. Queries that do not align with the schema or include assumptions will be considered invalid.
Return only sql query
        """
 
    @step
    async def start_chat(self, ev: StartEvent) -> StopEvent:
            question = ev.topic
            # """Process user input, generate SQL query, execute it, and visualize results."""
            schema = self.get_schema()
            prompt = self.engineer_prompt(question, schema)
   
            try:
                sql_query = self.vn.generate_sql(prompt)
                print(f"Generated SQL Query: {sql_query}")
   
                # Execute SQL query
                result_df = self.run_query(sql_query)
               
   
               
                # Anonymize data
                    # anonymized_df = self.anonymize_data(result_df)
   
                plotly_code = self.vn.generate_plotly_code(question=question, sql=sql_query, df=result_df)
                fig = self.vn.get_plotly_figure(plotly_code=plotly_code, df=result_df)
               
                return StopEvent(result = [result_df,fig])
   
   
            except Exception as e:
                result = "It seems the server is currently unavailable. Please try again later or resubmit your query."
                return StopEvent(result)
 
 
 
async def main():
    st.title("Athena Query Agent")
    class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, config=config)
    # Initialize the AthenaAgent
    with open('secret.key', 'rb') as key_file:
        key = key_file.read()

    cipher_suite = Fernet(key)

    # Load the encrypted configuration data
    with open('config.json', 'r') as config_file:
        encrypted_data = json.load(config_file)

    # Decrypt the sensitive information
    data = {key: cipher_suite.decrypt(value.encode()).decode() for key, value in encrypted_data.items()}
    vn = MyVanna(config={'api_key':data["API_KEY"] ,'model': 'gpt-3.5-turbo', 'temperature': 0.2,'path': 'embedding_aws_matrix2'})
    agent = AthenaAgent(vn)
   
    # User input
    question = st.text_input("Enter your query:")
 
 
 
    if st.button("Run Query") and question:
        with st.spinner("Processing..."):
            try:
 
                result = await agent.run(topic=question)  # Run async function
 
                if isinstance(result, list):
                    result_df, fig = result
                    st.write("Query Results:")
                    st.dataframe(result_df)
                    # st.write("Visualization:")
                    # st.plotly_chart(fig)
                else:
                    st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
 
 
 
if __name__ == "__main__":
    asyncio.run(main())  # Properly runs the async function