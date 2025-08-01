import streamlit as st
import asyncio
import httpx
import json
import datetime
import os
from typing import List, Dict, Any

# Set page title and configuration
st.set_page_config(
    page_title="GitHub Code Monitor Chat",
    page_icon="💬",
    layout="wide"
)

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000/api")  # Update with /api prefix

# Helper functions
async def get_repositories():
    """Fetch available repositories from the API"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{API_URL}/repositories")
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to fetch repositories: {response.text}")
                return []
        except httpx.ConnectError:
            st.error(f"Connection refused. Make sure the API server is running at {API_URL}.")
            return []
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")
            return []

async def send_chat_query(query: str, repository_id: int = None):
    """Send a chat query to the API"""
    # Format request according to ChatRequest schema
    payload = {
        "query": query,
        "repository_id": repository_id
    }
    
    st.info("Sending query to API...")
    
    # Increase timeout to 120 seconds for complex queries
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            st.session_state.connection_error = None
            
            # Add a retry mechanism for better reliability
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = await client.post(
                        f"{API_URL}/chat",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    break  # Exit retry loop if successful
                except (httpx.ConnectError, httpx.ReadTimeout) as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise  # Re-raise the last exception if all retries failed
                    await asyncio.sleep(1)  # Wait before retrying
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                error_msg = "API endpoint not found. Make sure the API server is running and the endpoint is correct."
                st.session_state.connection_error = error_msg
                st.error(error_msg)
                return {
                    "response": error_msg,
                    "context": {},
                    "references": []
                }
            else:
                error_msg = f"Failed to get response: {response.status_code} - {response.text}"
                st.session_state.connection_error = error_msg
                st.error(error_msg)
                return {
                    "response": f"Error: The API returned status code {response.status_code}. {response.text}",
                    "context": {},
                    "references": []
                }
        except httpx.ConnectError as e:
            error_msg = f"Error connecting to API: Connection refused. Make sure the API server is running at {API_URL}."
            st.session_state.connection_error = error_msg
            st.error(error_msg)
            return {
                "response": error_msg,
                "context": {},
                "references": []
            }
        except httpx.ReadTimeout as e:
            error_msg = f"API request timed out after 120s. The server might be processing a complex query or is overloaded."
            st.session_state.connection_error = error_msg
            st.error(error_msg)
            return {
                "response": error_msg,
                "context": {},
                "references": []
            }
        except Exception as e:
            error_msg = f"Error connecting to API: {str(e)}"
            st.session_state.connection_error = error_msg
            st.error(error_msg)
            return {
                "response": error_msg,
                "context": {},
                "references": []
            }

# UI Layout
st.title("💬 GitHub Code Monitor Chat")
st.markdown("""
Ask questions about your code repository, like:
- What was added in the last commit?
- Show me commits affecting the login flow
- Explain the changes to the auth module
""")

# Sidebar with repository selection
st.sidebar.header("Repository Selection")

# Get repositories
repos = asyncio.run(get_repositories())
repo_options = [{"id": None, "name": "All Repositories"}] + repos

selected_repo_index = st.sidebar.selectbox(
    "Select a repository:",
    range(len(repo_options)),
    format_func=lambda i: repo_options[i]["name"]
)

selected_repo_id = repo_options[selected_repo_index]["id"] if selected_repo_index > 0 else None

# Chat history management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show references if available
        if message["role"] == "assistant" and "references" in message and message["references"]:
            st.markdown("**References:**")
            for ref in message["references"]:
                st.markdown(f"- {ref['title']}")

# Chat input
if prompt := st.chat_input("Ask about your code repository..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = asyncio.run(send_chat_query(prompt, selected_repo_id))
            
            # Display the main response in a clean format
            st.markdown(response_data["response"])
            
            # Show code analysis if available
            if response_data.get("context", {}).get("code_analysis"):
                with st.expander("🧠 AI Code Analysis"):
                    st.markdown(response_data["context"]["code_analysis"])
            
            # Show relevant code snippets
            if response_data.get("context", {}).get("snippets"):
                with st.expander(f"📝 Relevant Code ({len(response_data['context']['snippets'])} snippets)"):
                    tabs = st.tabs([
                        f"{snippet.get('name', snippet.get('file_path', f'Snippet {i}'))}" 
                        for i, snippet in enumerate(response_data["context"]["snippets"][:5])
                    ])
                    
                    for i, (tab, snippet) in enumerate(zip(tabs, response_data["context"]["snippets"][:5])):
                        with tab:
                            language = snippet.get("language", "").lower() if snippet.get("language") else None
                            content = snippet.get("content", "")
                            
                            # Show snippet metadata
                            if snippet.get("item_type") == "snippet":
                                st.markdown(f"**Type:** {snippet.get('snippet_type', 'Code Snippet')}")
                                if snippet.get("start_line") and snippet.get("end_line"):
                                    st.markdown(f"**Lines:** {snippet.get('start_line')} - {snippet.get('end_line')}")
                            elif snippet.get("item_type") == "file":
                                st.markdown(f"**File:** {snippet.get('file_path', 'Unknown')}")
                                if snippet.get("commit_message"):
                                    st.markdown(f"**From commit:** {snippet.get('commit_message')}")
                            
                            # Show the code with proper formatting
                            st.code(content, language=language)
                            
                            # Show similarity score if available
                            if snippet.get("similarity") is not None:
                                st.progress(snippet.get("similarity"), text=f"Relevance: {snippet.get('similarity'):.2f}")
            
            # Show detailed commit information in a structured way
            if response_data.get("context", {}).get("stats", {}).get("commits"):
                with st.expander("🔍 Commit Information"):
                    for commit in response_data["context"]["stats"]["commits"]:
                        st.markdown("---")
                        
                        # Commit header with hash and message
                        st.markdown(f"### Commit {commit['commit_hash'][:7]}")
                        st.info(commit['message'])
                        
                        # Metadata in columns
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Author:** {commit['author']}")
                            st.markdown(f"**Date:** {commit.get('date', '')}")
                        
                        # File changes section if available
                        if commit.get('files'):
                            st.markdown("#### Files Changed")
                            
                            # Create a table view of changed files
                            import pandas as pd
                            files_data = []
                            for file in commit['files']:
                                files_data.append({
                                    'File': file['file_path'],
                                    'Change Type': file['change_type'].capitalize(),
                                    'Language': file['language'] or 'Unknown'
                                })
                            if files_data:
                                df = pd.DataFrame(files_data)
                                st.dataframe(df, hide_index=True, use_container_width=True)
    
    # Add response to chat history with enhanced references
    references = []
    if response_data.get("references"):
        references = response_data["references"]
    elif response_data.get("context", {}).get("snippets"):
        # Create references from snippets if not explicitly provided
        for snippet in response_data["context"]["snippets"]:
            name = snippet.get("name", snippet.get("file_path", f"Snippet {snippet.get('id', '')}"))
            description = ""
            
            if snippet.get("item_type") == "snippet":
                description = f"Lines {snippet.get('start_line', '?')}-{snippet.get('end_line', '?')}"
            elif snippet.get("item_type") == "file" and snippet.get("commit_message"):
                description = f"From commit: {snippet.get('commit_message', '')[:50]}"
                
            references.append({
                "id": str(snippet.get("id", "")),
                "title": f"{name} ({snippet.get('item_type', 'code')})",
                "description": description
            })
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_data["response"],
        "references": references
    })

# Display additional information in the sidebar
st.sidebar.header("Additional Information")

if selected_repo_id is not None:
    st.sidebar.subheader("Repository Info")
    for repo in repos:
        if repo["id"] == selected_repo_id:
            st.sidebar.write(f"**Name:** {repo['name']}")
            st.sidebar.write(f"**Owner:** {repo['owner']}")
            if repo.get("description"):
                st.sidebar.write(f"**Description:** {repo['description']}")
            break

st.sidebar.subheader("About")
st.sidebar.info("""
This chat interface allows you to interact with your GitHub code repositories.
Ask questions about commits, files, or code changes to get AI-powered insights.
""")

# Footer
st.markdown("---")
st.markdown("GitHub Code Monitor - Powered by AI")