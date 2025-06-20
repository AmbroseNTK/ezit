import cv2
import numpy as np
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import base64
from io import BytesIO
from PIL import Image
import openai
import os

class Node:
    """Represents a processing node in the workflow"""
    
    def __init__(self, node_id: str, node_type: str, parameters: Optional[Dict[str, Any]] = None, 
                 inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.parameters = parameters or {}
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.result = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "parameters": self.parameters,
            "inputs": self.inputs,
            "outputs": self.outputs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create node from dictionary representation"""
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            parameters=data.get("parameters", {}),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", [])
        )

class WorkflowProcessor:
    """Main workflow processor for image processing"""
    
    def __init__(self):
        self.nodes = {}
        self.connections = []
        self.node_specs = self._load_node_specifications()
        self.execution_order = []
        
    def _load_node_specifications(self) -> Dict[str, Any]:
        """Load node specifications from node_spec.txt"""
        specs = {}
        
        with open('node_spec.txt', 'r') as f:
            content = f.read()
            
        # Parse the specifications using regex
        node_blocks = re.split(r'### \d+\. ', content)[1:]  # Skip the header
        
        for block in node_blocks:
            lines = block.strip().split('\n')
            # Extract node name from the first line - look for the name field
            node_name = None
            for line in lines:
                if line.strip().startswith('- **name**: '):
                    node_name = line.split('"')[1]  # Extract name between quotes
                    break
            
            if not node_name:
                continue  # Skip if no name found
                
            spec = {
                "name": node_name,
                "description": "",
                "parameters": {},
                "inputs": [],
                "outputs": [],
                "operator": ""
            }
            
            for line in lines[1:]:
                line = line.strip()
                if line.startswith("- **description**: "):
                    spec["description"] = line.split(": ", 1)[1].strip('"')
                elif line.startswith("- **parameters**: "):
                    # Parse parameters (simplified parsing)
                    param_text = line.split(": ", 1)[1]
                    if param_text != "{}":
                        # Extract parameter definitions
                        spec["parameters"] = self._parse_parameters(param_text)
                elif line.startswith("- **inputs**: "):
                    inputs_text = line.split(": ", 1)[1]
                    if inputs_text != "[]":
                        spec["inputs"] = [s.strip('"') for s in inputs_text.strip('[]').split(',')]
                elif line.startswith("- **outputs**: "):
                    outputs_text = line.split(": ", 1)[1]
                    if outputs_text != "[]":
                        spec["outputs"] = [s.strip('"') for s in outputs_text.strip('[]').split(',')]
                elif line.startswith("- **operator**: "):
                    spec["operator"] = line.split(": ", 1)[1].strip('"')
            
            specs[node_name] = spec
            
        # Debug output
        print(f"Loaded node specifications: {list(specs.keys())}")
            
        return specs
    
    def _parse_parameters(self, param_text: str) -> Dict[str, Any]:
        """Parse parameters from text representation"""
        params = {}
        # This is a simplified parser - in production you'd want a more robust solution
        try:
            # Extract parameter definitions using regex
            param_matches = re.findall(r'"([^"]+)":\s*{([^}]+)}', param_text)
            for param_name, param_def in param_matches:
                type_match = re.search(r'"type":\s*"([^"]+)"', param_def)
                default_match = re.search(r'"default":\s*([^,}]+)', param_def)
                desc_match = re.search(r'"description":\s*"([^"]+)"', param_def)
                
                if type_match:
                    param_type = type_match.group(1)
                    default_val = None
                    if default_match:
                        default_str = default_match.group(1).strip()
                        if default_str.startswith('[') and default_str.endswith(']'):
                            # Parse tuple/list
                            default_val = [int(x.strip()) for x in default_str.strip('[]').split(',')]
                        elif default_str.isdigit():
                            default_val = int(default_str)
                        elif default_str.replace('.', '').replace('-', '').isdigit():
                            default_val = float(default_str)
                        elif default_str.startswith('"') and default_str.endswith('"'):
                            default_val = default_str.strip('"')
                        else:
                            default_val = default_str
                    
                    params[param_name] = {
                        "type": param_type,
                        "default": default_val,
                        "description": desc_match.group(1) if desc_match else ""
                    }
        except Exception as e:
            print(f"Warning: Could not parse parameters: {e}")
            
        return params
    
    def add_node(self, node: Node) -> None:
        """Add a node to the workflow"""
        self.nodes[node.node_id] = node
        
    def add_connection(self, from_node: str, to_node: str, from_output: str = "image", to_input: str = "image") -> None:
        """Add a connection between nodes"""
        self.connections.append({
            "from_node": from_node,
            "to_node": to_node,
            "from_output": from_output,
            "to_input": to_input
        })
        
    def _determine_execution_order(self) -> List[str]:
        """Determine the topological order of node execution"""
        # Build adjacency list
        graph = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}
        
        for conn in self.connections:
            graph[conn["from_node"]].append(conn["to_node"])
            in_degree[conn["to_node"]] += 1
            
        # Topological sort using Kahn's algorithm
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(order) != len(self.nodes):
            raise ValueError("Workflow contains cycles or disconnected nodes")
            
        return order
    
    def _execute_node(self, node: Node, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single node with given input data"""
        operator = self.node_specs[node.node_type]["operator"]
        
        print(f"Executing node {node.node_id} of type {node.node_type} with operator {operator}")
        
        if operator == "input":
            return {"image": input_data.get("image")}
            
        elif operator == "output":
            return {"image": input_data.get("image")}
            
        elif operator == "gaussian_blur":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for gaussian_blur operation")
            kernel_size = tuple(node.parameters.get("kernel_size", [5, 5]))
            sigma_x = node.parameters.get("sigma_x", 0.0)
            sigma_y = node.parameters.get("sigma_y", 0.0)
            result = cv2.GaussianBlur(image, kernel_size, sigma_x, sigma_y)
            return {"image": result}
            
        elif operator == "median_blur":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for median_blur operation")
            kernel_size = node.parameters.get("kernel_size", 5)
            result = cv2.medianBlur(image, kernel_size)
            return {"image": result}
            
        elif operator == "bilateral_filter":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for bilateral_filter operation")
            d = node.parameters.get("d", 15)
            sigma_color = node.parameters.get("sigma_color", 75.0)
            sigma_space = node.parameters.get("sigma_space", 75.0)
            result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            return {"image": result}
            
        elif operator == "rgb2gray":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for rgb2gray operation")
            if len(image.shape) == 3:
                result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                result = image
            return {"image": result}
            
        elif operator == "gray2rgb":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for gray2rgb operation")
            if len(image.shape) == 2:
                result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                result = image
            return {"image": result}
            
        elif operator == "canny_edge":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for canny_edge operation")
            threshold1 = node.parameters.get("threshold1", 100.0)
            threshold2 = node.parameters.get("threshold2", 200.0)
            aperture_size = node.parameters.get("aperture_size", 3)
            result = cv2.Canny(image, threshold1, threshold2, apertureSize=aperture_size)
            return {"image": result}
            
        elif operator == "sobel_edge":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for sobel_edge operation")
            dx = node.parameters.get("dx", 1)
            dy = node.parameters.get("dy", 1)
            ksize = node.parameters.get("ksize", 3)
            scale = node.parameters.get("scale", 1.0)
            delta = node.parameters.get("delta", 0.0)
            result = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=ksize, scale=scale, delta=delta)
            result = np.uint8(np.absolute(result))
            return {"image": result}
            
        elif operator == "threshold":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for threshold operation")
            thresh = node.parameters.get("thresh", 127.0)
            maxval = node.parameters.get("maxval", 255.0)
            thresh_type = node.parameters.get("type", "THRESH_BINARY")
            
            # Convert string to cv2 constant
            if thresh_type == "THRESH_BINARY":
                thresh_type = cv2.THRESH_BINARY
            elif thresh_type == "THRESH_BINARY_INV":
                thresh_type = cv2.THRESH_BINARY_INV
            elif thresh_type == "THRESH_TRUNC":
                thresh_type = cv2.THRESH_TRUNC
            elif thresh_type == "THRESH_TOZERO":
                thresh_type = cv2.THRESH_TOZERO
            elif thresh_type == "THRESH_TOZERO_INV":
                thresh_type = cv2.THRESH_TOZERO_INV
                
            _, result = cv2.threshold(image, thresh, maxval, thresh_type)
            return {"image": result}
            
        elif operator == "adaptive_threshold":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for adaptive_threshold operation")
            max_value = node.parameters.get("max_value", 255.0)
            adaptive_method = node.parameters.get("adaptive_method", "ADAPTIVE_THRESH_GAUSSIAN_C")
            threshold_type = node.parameters.get("threshold_type", "THRESH_BINARY")
            block_size = node.parameters.get("block_size", 11)
            c = node.parameters.get("c", 2.0)
            
            # Convert strings to cv2 constants
            if adaptive_method == "ADAPTIVE_THRESH_GAUSSIAN_C":
                adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            elif adaptive_method == "ADAPTIVE_THRESH_MEAN_C":
                adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
                
            if threshold_type == "THRESH_BINARY":
                threshold_type = cv2.THRESH_BINARY
            elif threshold_type == "THRESH_BINARY_INV":
                threshold_type = cv2.THRESH_BINARY_INV
                
            result = cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, c)
            return {"image": result}
            
        elif operator == "morphology":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for morphology operation")
            operation = node.parameters.get("operation", "MORPH_OPEN")
            kernel_size = node.parameters.get("kernel_size", 3)
            iterations = node.parameters.get("iterations", 1)
            
            # Create kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            
            # Convert string to cv2 constant
            if operation == "MORPH_OPEN":
                operation = cv2.MORPH_OPEN
            elif operation == "MORPH_CLOSE":
                operation = cv2.MORPH_CLOSE
            elif operation == "MORPH_ERODE":
                operation = cv2.MORPH_ERODE
            elif operation == "MORPH_DILATE":
                operation = cv2.MORPH_DILATE
                
            result = cv2.morphologyEx(image, operation, kernel, iterations=iterations)
            return {"image": result}
            
        elif operator == "histogram_equalization":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for histogram_equalization operation")
            if len(image.shape) == 2:
                result = cv2.equalizeHist(image)
            else:
                # For color images, convert to YUV and equalize Y channel
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return {"image": result}
            
        elif operator == "clahe":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for clahe operation")
            clip_limit = node.parameters.get("clip_limit", 2.0)
            tile_grid_size = tuple(node.parameters.get("tile_grid_size", [8, 8]))
            
            if len(image.shape) == 2:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                result = clahe.apply(image)
            else:
                # For color images, apply to each channel
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                yuv[:,:,0] = clahe.apply(yuv[:,:,0])
                result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return {"image": result}
            
        elif operator == "color_space_convert":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for color_space_convert operation")
            code = node.parameters.get("code", "COLOR_BGR2HSV")
            
            # Convert string to cv2 constant
            if code == "COLOR_BGR2HSV":
                code = cv2.COLOR_BGR2HSV
            elif code == "COLOR_HSV2BGR":
                code = cv2.COLOR_HSV2BGR
            elif code == "COLOR_BGR2GRAY":
                code = cv2.COLOR_BGR2GRAY
            elif code == "COLOR_GRAY2BGR":
                code = cv2.COLOR_GRAY2BGR
            elif code == "COLOR_BGR2LAB":
                code = cv2.COLOR_BGR2LAB
            elif code == "COLOR_LAB2BGR":
                code = cv2.COLOR_LAB2BGR
                
            result = cv2.cvtColor(image, code)
            return {"image": result}
            
        elif operator == "resize":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for resize operation")
            width = node.parameters.get("width", 640)
            height = node.parameters.get("height", 480)
            interpolation = node.parameters.get("interpolation", "INTER_LINEAR")
            
            # Convert string to cv2 constant
            if interpolation == "INTER_LINEAR":
                interpolation = cv2.INTER_LINEAR
            elif interpolation == "INTER_NEAREST":
                interpolation = cv2.INTER_NEAREST
            elif interpolation == "INTER_CUBIC":
                interpolation = cv2.INTER_CUBIC
            elif interpolation == "INTER_AREA":
                interpolation = cv2.INTER_AREA
                
            result = cv2.resize(image, (width, height), interpolation=interpolation)
            return {"image": result}
            
        elif operator == "crop":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for crop operation")
            x = node.parameters.get("x", 0)
            y = node.parameters.get("y", 0)
            width = node.parameters.get("width", 100)
            height = node.parameters.get("height", 100)
            
            result = image[y:y+height, x:x+width]
            return {"image": result}
            
        elif operator == "brightness_contrast":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for brightness_contrast operation")
            alpha = node.parameters.get("alpha", 1.0)  # Contrast
            beta = node.parameters.get("beta", 0.0)    # Brightness
            
            result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return {"image": result}
            
        elif operator == "gamma_correction":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for gamma_correction operation")
            gamma = node.parameters.get("gamma", 1.0)
            
            # Build lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            result = cv2.LUT(image, table)
            return {"image": result}
            
        elif operator == "add_noise":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for add_noise operation")
            mean = node.parameters.get("mean", 0.0)
            std = node.parameters.get("std", 25.0)
            
            noise = np.random.normal(mean, std, image.shape).astype(np.float32)
            result = cv2.add(image.astype(np.float32), noise)
            result = np.clip(result, 0, 255).astype(np.uint8)
            return {"image": result}
            
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def execute_workflow(self, input_image: np.ndarray) -> np.ndarray:
        """Execute the complete workflow with input image"""
        if not self.nodes:
            raise ValueError("No nodes in workflow")
            
        # Determine execution order
        self.execution_order = self._determine_execution_order()
        
        # Initialize node results
        node_results = {}
        
        # Execute nodes in order
        for node_id in self.execution_order:
            node = self.nodes[node_id]
            
            # Collect inputs for this node
            input_data = {}
            for conn in self.connections:
                if conn["to_node"] == node_id:
                    from_node_id = conn["from_node"]
                    if from_node_id in node_results:
                        input_data[conn["to_input"]] = node_results[from_node_id][conn["from_output"]]
            
            # Add input image for input nodes
            if node.node_type == "input":
                input_data["image"] = input_image
            
            # Execute node
            result = self._execute_node(node, input_data)
            node_results[node_id] = result
            node.result = result
        
        # Find output node
        output_node = None
        for node in self.nodes.values():
            if node.node_type == "output":
                output_node = node
                break
                
        if output_node and output_node.node_id in node_results:
            return node_results[output_node.node_id]["image"]
        else:
            # Return result from last processing node
            for node_id in reversed(self.execution_order):
                node = self.nodes[node_id]
                if node.node_type not in ["input", "output"] and node_id in node_results:
                    return node_results[node_id]["image"]
                    
        raise ValueError("No output found in workflow")

class AIWorkflowGenerator:
    """AI-powered workflow generator using OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
        self.processor = WorkflowProcessor()
        
    def generate_workflow_from_prompt(self, image: np.ndarray, prompt: str) -> Dict[str, Any]:
        """Generate a workflow from a natural language prompt"""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
            
        # Convert image to base64 for API
        image_base64 = self._image_to_base64(image)
        
        # Read node specifications
        with open('node_spec.txt', 'r') as f:
            node_specs = f.read()
        
        # Create system prompt
        system_prompt = f"""You are an expert in computer vision and image processing. You need to create a workflow of image processing operations based on a user's request.

Available node specifications:
{node_specs}

Your task is to:
1. Analyze the user's request
2. Create a JSON workflow that describes the nodes and their connections
3. Choose appropriate parameters for each node
4. Ensure the workflow is valid and executable

The workflow should be returned as a JSON object with this structure:
{{
    "nodes": [
        {{
            "node_id": "unique_id",
            "node_type": "node_name",
            "parameters": {{}},
            "inputs": [],
            "outputs": []
        }}
    ],
    "connections": [
        {{
            "from_node": "node_id",
            "to_node": "node_id",
            "from_output": "image",
            "to_input": "image"
        }}
    ]
}}

Rules:
- Always start with an "input" node
- Always end with an "output" node
- Connect nodes sequentially through their inputs/outputs
- Use appropriate parameters for each operation
- Keep the workflow simple and focused on the user's request
- Ensure all connections are valid (image to image)"""
        
        # Create user prompt
        user_prompt = f"""Please create an image processing workflow for the following request:

User Request: {prompt}

The input image has dimensions: {image.shape[1]}x{image.shape[0]} pixels
Image type: {'Grayscale' if len(image.shape) == 2 else 'Color'} ({image.shape[2] if len(image.shape) == 3 else 1} channels)

Please generate a JSON workflow that will process this image according to the user's request."""
        
        try:
            # Call OpenAI API
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Extract and parse JSON from response
            response_text = response.choices[0].message.content

            print(response_text)

            if response_text is None:
                raise ValueError("No content received from OpenAI API")
            workflow_json = self._extract_json_from_response(response_text)
            
            print(f"Generated workflow JSON: {workflow_json}")
            
            return workflow_json
            
        except Exception as e:
            raise Exception(f"Failed to generate workflow: {str(e)}")
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Convert to PIL Image and then to base64
        pil_image = Image.fromarray(image_rgb)
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from OpenAI response"""
        # Try to find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in response")
            
        json_str = response_text[json_start:json_end]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")
    
    def create_workflow_from_json(self, workflow_json: Dict[str, Any]) -> WorkflowProcessor:
        """Create a WorkflowProcessor instance from JSON workflow"""
        processor = WorkflowProcessor()
        
        # Add nodes
        for node_data in workflow_json.get("nodes", []):
            node = Node.from_dict(node_data)
            processor.add_node(node)
        
        # Add connections
        for conn_data in workflow_json.get("connections", []):
            processor.add_connection(
                conn_data["from_node"],
                conn_data["to_node"],
                conn_data.get("from_output", "image"),
                conn_data.get("to_input", "image")
            )
        
        return processor

def process_image_with_ai_workflow(image: np.ndarray, prompt: str, api_key: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Main function to process image with AI-generated workflow"""
    generator = AIWorkflowGenerator(api_key)
    
    # Generate workflow
    workflow_json = generator.generate_workflow_from_prompt(image, prompt)
    
    # Create and execute workflow
    processor = generator.create_workflow_from_json(workflow_json)
    result_image = processor.execute_workflow(image)
    
    return result_image, workflow_json 