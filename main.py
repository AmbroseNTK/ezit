import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
from workflow_processor import process_image_with_ai_workflow

# Try to import streamlit-agraph, fallback to simple visualization if not available
try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGGRAPH_AVAILABLE = True
except ImportError:
    AGGRAPH_AVAILABLE = False
    st.warning("streamlit-agraph not available. Using simple workflow visualization.")

st.set_page_config(page_title="AI Image Workflow Processor", layout="wide")
st.title("🖼️ AI Image Workflow Processor")

st.markdown("""
Upload an image, describe what you want to do in natural language, and let AI build and execute an OpenCV workflow for you!
""")

# API Key input (hidden or via env)
def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API Key", type="password")
    return api_key

def create_simple_workflow_diagram(workflow_json):
    """Create a simple text-based workflow diagram"""
    if not workflow_json or 'nodes' not in workflow_json:
        return "No workflow data available"
    
    diagram = "```mermaid\ngraph TD\n"
    
    # Add nodes
    for node in workflow_json['nodes']:
        node_id = node['node_id']
        node_type = node['node_type']
        
        # Style nodes based on type
        if node_type == 'input':
            diagram += f"    {node_id}[{node_type}]:::input\n"
        elif node_type == 'output':
            diagram += f"    {node_id}[{node_type}]:::output\n"
        else:
            diagram += f"    {node_id}[{node_type}]:::process\n"
    
    # Add connections
    for conn in workflow_json.get('connections', []):
        diagram += f"    {conn['from_node']} --> {conn['to_node']}\n"
    
    # Add CSS classes
    diagram += "    classDef input fill:#4CAF50,stroke:#2E7D32,stroke-width:2px,color:#fff\n"
    diagram += "    classDef output fill:#F44336,stroke:#C62828,stroke-width:2px,color:#fff\n"
    diagram += "    classDef process fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff\n"
    diagram += "```"
    
    return diagram

def create_workflow_diagram(workflow_json):
    """Create a visual diagram of the workflow"""
    if not workflow_json or 'nodes' not in workflow_json:
        return None
    
    # Create nodes for the graph
    nodes = []
    for node_data in workflow_json['nodes']:
        node_id = node_data['node_id']
        node_type = node_data['node_type']
        
        # Define colors based on node type
        if node_type == 'input':
            color = '#4CAF50'  # Green
        elif node_type == 'output':
            color = '#F44336'  # Red
        else:
            color = '#2196F3'  # Blue
        
        # Create node with label
        node = Node(
            id=node_id,
            label=f"{node_type}\n({node_id})",
            size=25,
            color=color,
            font={"size": 12}
        )
        nodes.append(node)
    
    # Create edges for the graph
    edges = []
    for conn in workflow_json.get('connections', []):
        edge = Edge(
            source=conn['from_node'],
            target=conn['to_node'],
            type="CURVE_SMOOTH",
            animated=True
        )
        edges.append(edge)
    
    return nodes, edges

def display_workflow_visualization(workflow_json):
    """Display the workflow as an interactive diagram"""
    if not workflow_json:
        return
    
    st.subheader("📊 Workflow Visualization")
    
    if AGGRAPH_AVAILABLE:
        # Use streamlit-agraph for interactive visualization
        diagram_data = create_workflow_diagram(workflow_json)
        if diagram_data:
            nodes, edges = diagram_data
            
            # Configure the graph
            config = Config(
                height=400,
                width=800,
                directed=True,
                physics=True,
                hierarchical=True,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=True,
                node={'labelProperty': 'label'},
                link={'labelProperty': 'label', 'renderLabel': True}
            )
            
            # Display the graph
            agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.warning("Could not create workflow visualization")
    else:
        # Use simple text-based diagram
        diagram = create_simple_workflow_diagram(workflow_json)
        st.markdown(diagram)
        
        # Also show a simple flow representation
        st.write("**Workflow Flow:**")
        flow_text = ""
        for i, node in enumerate(workflow_json.get('nodes', []), 1):
            flow_text += f"{i}. {node['node_type']} ({node['node_id']})\n"
        st.text(flow_text)

def display_workflow_text(workflow_json):
    """Display workflow as formatted text"""
    if not workflow_json:
        return
    
    st.subheader("📋 Workflow Details")
    
    # Display nodes
    st.write("**Nodes:**")
    for i, node in enumerate(workflow_json.get('nodes', []), 1):
        with st.expander(f"{i}. {node['node_type']} ({node['node_id']})"):
            st.json(node)
    
    # Display connections
    st.write("**Connections:**")
    for i, conn in enumerate(workflow_json.get('connections', []), 1):
        st.write(f"{i}. {conn['from_node']} → {conn['to_node']}")

api_key = get_api_key()

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    prompt = st.text_area("Describe your image processing workflow (e.g. 'Convert to grayscale and blur the image'):")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    if image_np.ndim == 2:
        st.image(image, caption="Uploaded Image (Grayscale)", use_column_width=True)
    else:
        st.image(image, caption="Uploaded Image", use_column_width=True)
else:
    image_np = None

# Process button
if st.button("Generate & Run Workflow", disabled=(uploaded_file is None or not prompt or not api_key)):
    if image_np is not None and prompt and api_key:
        with st.spinner("Generating workflow and processing image..."):
            try:
                # Convert RGBA to RGB if needed
                if image_np.shape[-1] == 4:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
                elif image_np.shape[-1] == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                result_image, workflow_json = process_image_with_ai_workflow(image_np, prompt, api_key)
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📸 Result Image")
                    # Convert result to displayable format
                    if result_image.ndim == 2:
                        st.image(result_image, caption="Result Image (Grayscale)", use_column_width=True)
                    else:
                        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption="Result Image", use_column_width=True)
                
                with col2:
                    st.subheader("🔄 Generated Workflow")
                    st.json(workflow_json)
                
                # Display workflow visualization
                display_workflow_visualization(workflow_json)
                
                # Display detailed workflow information
                display_workflow_text(workflow_json)
                
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please upload an image, enter a prompt, and provide your OpenAI API key.")
