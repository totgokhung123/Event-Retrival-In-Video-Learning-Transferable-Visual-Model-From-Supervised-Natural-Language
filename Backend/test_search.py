import os
from services import initialize_services

# Initialize services
BASE_DIR = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\static\\processed_frames"
METADATA_DIR = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\metadata"
EMBEDDING_DIR = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\embedding"
MODEL_DIR = "E:\\Đồ án tôt nghiệp\\source_code\\Backend\\models"
FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, "final_checkpoint.pt")

# Open a file for writing results
with open("search_test_results.txt", "w") as f:
    f.write("Initializing services...\n")
    service_container = initialize_services(BASE_DIR, METADATA_DIR, EMBEDDING_DIR, FINETUNED_MODEL_PATH)
    embedding_service = service_container['embedding_service']
    search_service = service_container['search_service']

    # Test search functionality
    f.write("\nTesting search_top_frames:\n")
    results = embedding_service.search_top_frames("person", 5)
    f.write(f"Search results: {results}\n")

    f.write("\nTesting search_semantic_with_clip:\n")
    results = search_service.search_semantic_with_clip("person", 0.5, 5)
    f.write(f"Search results count: {len(results)}\n")
    if results:
        f.write("Sample result confidences:\n")
        for i, result in enumerate(results[:3]):
            f.write(f"{i+1}. Confidence: {result.get('confidence')}, Clip similarity: {result.get('clip_similarity')}\n")

    f.write("\nTest completed!\n") 