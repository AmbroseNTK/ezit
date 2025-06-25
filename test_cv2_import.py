#!/usr/bin/env python3
"""
Test script to verify cv2 import and functionality in workflow_processor
"""

import numpy as np
from workflow_processor import WorkflowProcessor, Node

def test_cv2_functionality():
    """Test that cv2 functions work correctly"""
    print("Testing cv2 functionality...")
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Create a simple workflow
    processor = WorkflowProcessor()
    
    # Add input node
    input_node = Node("input", "input")
    processor.add_node(input_node)
    
    # Add a gaussian blur node
    blur_node = Node("blur", "gaussian_blur", {"kernel_size": [5, 5]})
    processor.add_node(blur_node)
    
    # Add output node
    output_node = Node("output", "output")
    processor.add_node(output_node)
    
    # Add connections
    processor.add_connection("input", "blur")
    processor.add_connection("blur", "output")
    
    # Execute workflow
    try:
        result = processor.execute_workflow(test_image)
        print(f"✅ Workflow executed successfully!")
        print(f"   Input shape: {test_image.shape}")
        print(f"   Output shape: {result.shape}")
        return True
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        return False

if __name__ == "__main__":
    success = test_cv2_functionality()
    if success:
        print("\n🎉 All tests passed! cv2 is working correctly.")
    else:
        print("\n💥 Tests failed. There may be an issue with cv2.") 