"""Configuration for GraphDiffLayout experiments."""

EXPERIMENT_PARAMS = {
    'image_size': (256, 256),
    'random_seed': 42,
    
    'layout_text_threshold': 50,  # Maximum distance for edge creation
    
    'small_object_size': 20,  # Maximum size to be considered a small object
    
    'object_counts': [5, 10, 20, 50],  # Number of objects for scalability test
}

VIZ_PARAMS = {
    'figsize': (10, 6),
    'dpi': 300,  # High quality for academic papers
    'format': 'pdf',
}
