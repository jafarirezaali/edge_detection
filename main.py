from edge_detection import EdgeDetection

# Example usage
if __name__ == "__main__":
    ed = EdgeDetection()
    blurred = ed.make_blurred("test.png")
    sobX, sobY, G = ed.sobel(blurred)
    grad_dir = ed.sobel_with_gradent(sobX, sobY)
    quantized = ed.quantize_angles(grad_dir)
    nms_result = ed.non_maximum_suppression(G, quantized)
    edges = ed.canny(nms_result)